from sampler import create_sampler, clean_sampler
import pyprind
import scipy
import numpy
from lightfm import LightFM
from utils import dict_to_coo
import theano
import time, sys
import theano.tensor as T
from theano.ifelse import ifelse

print "Reload?"


class MLP(object):
    def __init__(self, feature_dim, n_embedding_dim, usage_count,
                 activation="T.nnet.relu", n_layers=2, width=128,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.0000):
        self.lambda_weight_l1 = lambda_weight_l1
        self.lambda_weight_l2 = lambda_weight_l2
        self.n_embedding_dim = n_embedding_dim
        self.n_layers = n_layers
        self.usage_count = usage_count
        self.width = width
        # create weights
        raw_visual_feature_factors = feature_dim
        self.activation = activation
        self.weights = []
        self.bias = []
        for i in range(self.n_layers):
            in_dim = self.width
            out_dim = self.width
            if i == 0:
                in_dim = raw_visual_feature_factors
            if i == self.n_layers - 1:
                out_dim = n_embedding_dim
            self.weights += [
                theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (in_dim + out_dim)),
                        high=numpy.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True, name='weights_' + str(i)),
            ]
            self.bias += [theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (in_dim + out_dim)),
                    high=numpy.sqrt(6. / (in_dim + out_dim)),
                    size=(out_dim,)
                ),
                dtype=theano.config.floatX
            ), borrow=True, name='weight_bias_' + str(i))]

    def projection(self, features, scale, max_norm=numpy.Infinity, training=False, dropout_rate=0.5):
        from theano.tensor.shared_randomstreams import RandomStreams
        import theano.sparse
        srng = RandomStreams(seed=12345)
        def projection_f(j):
            activation_fn = eval(self.activation)
            ret = features #features[j]
            for i in range(self.n_layers - 1):
                hidden1 = self.weights[i]
                if training:
                    hidden1 = T.switch(srng.binomial(size=hidden1.shape, p=dropout_rate), hidden1, 0)
                else:
                    hidden1 = dropout_rate * hidden1
                dot = T.dot
                if i == 0 and isinstance(features.type, theano.sparse.SparseType):
                    dot = theano.sparse.dot
                ret = activation_fn(dot(ret, hidden1) + self.bias[i])
            dot = T.dot
            if self.n_layers == 1 and isinstance(features.type, theano.sparse.SparseType):
                dot = theano.sparse.dot
            embedding = dot(ret, self.weights[-1]) + self.bias[-1]
            embedding *= scale
            row_norms = T.sqrt(T.sum(T.sqr(embedding), axis=1))
            desired_norms = T.clip(row_norms, 0, max_norm)
            embedding *= (desired_norms / (row_norms )).reshape((embedding.shape[0], 1))
            return embedding
        return projection_f

    def l2_reg(self):
        return [[w, self.lambda_weight_l2, ] for w in self.weights] + \
               [[w, self.lambda_weight_l2, ] for w in self.bias]

    def l1_reg(self):
        return [[w, self.lambda_weight_l1, ] for w in self.weights]

    def updates(self):
        updates = []
        updates += [
            [w, self.usage_count]
            for w in self.weights]
        updates += [
            [w, self.usage_count]
            for w in self.bias]
        return updates

    def copy(self):
        import theano.misc.pkl_utils
        buf = theano.misc.pkl_utils.BytesIO()
        theano.misc.pkl_utils.dump(self, buf)
        return theano.misc.pkl_utils.load(buf)


class Model(object):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.signature = None

    def check_signature(self, train_dict):
        signature = sum(map(lambda u: sum(u[1])%(u[0]+1), train_dict.items()))
        if self.signature is None or self.signature == signature:
            self.signature = signature
            return True
        raise Exception("Inconsistent train dict signature")

    def normalize(self, variable):
        return variable / ((T.sum(variable ** 2, axis=1) ** 0.5).reshape((variable.shape[0], 1)))

    def copy(self):
        import theano.misc.pkl_utils
        buf = theano.misc.pkl_utils.BytesIO()
        theano.misc.pkl_utils.dump(self, buf)
        return theano.misc.pkl_utils.load(buf)

    def params(self):
        return [
            ["U", self.n_users],
            ["V", self.n_items]
        ]

    def scores_for_users(self, users):
        return numpy.zeros((len(users), self.n_items))

    def auc(self, likes_dict, exclude_dict):
        sequence = numpy.arange(self.n_items)

        def to_ranks(order):
            r = numpy.empty(self.n_items, int)
            r[order] = sequence
            return r

        aucs = []
        try:
            users = likes_dict.keys()
            bar = pyprind.ProgBar(len(users))
            for inx, scores in enumerate(self.scores_for_users(users)):
                user = users[inx]
                # here the rank is larger if we predict it is more preferable
                ranks = to_ranks(numpy.argsort(scores))
                excludes = []
                if exclude_dict is not None and user in exclude_dict:
                    excludes = list(exclude_dict[user])
                likes = list(likes_dict[user])
                pos_count = len(likes)
                neg_count = self.n_items - pos_count - len(excludes)
                total = pos_count * neg_count
                exclude_ranks = ranks[excludes]
                like_ranks = ranks[likes]
                hit = numpy.sum(like_ranks)
                hit -= sum([1 for x in like_ranks for y in exclude_ranks if x > y])
                hit -= (len(likes) ** 2 - len(likes)) / 2
                aucs.append(hit * 1.0 / total)
                bar.update(item_id=str(numpy.mean(aucs)))
        except Exception as e:
            sys.stderr.write(str(e))
        finally:
            return numpy.mean(aucs), scipy.stats.sem(aucs)

    def topN(self, users, exclude_dict, n=100, exclude_items=None):
        # return top N item (sorted)
        for inx, scores in enumerate(self.scores_for_users(users)):
            user = users[inx]
            if exclude_dict is not None and user in exclude_dict:
                scores[list(exclude_dict[user])] = -numpy.Infinity
            if exclude_items is not None and len(exclude_items) > 0:
                scores[exclude_items] = -numpy.Infinity
            top = numpy.argpartition(-scores, n)[0:n]
            yield sorted(top, key=lambda i: -scores[i])

    def recall(self, likes_dict, exclude_dict, ns=(100, 50, 10), n_users=None, exclude_items=None, users=None):
        from collections import defaultdict
        recall = defaultdict(list)

        if users is None:
            if n_users is None:
                users = likes_dict.keys()
            else:
                numpy.random.seed(1)
                users = numpy.random.choice(likes_dict.keys(), replace=False, size=n_users)
        bar = pyprind.ProgBar(len(users))
        for inx, top in enumerate(self.topN(users, exclude_dict, n=max(ns), exclude_items=exclude_items)):
            user = users[inx]
            likes = likes_dict[user]
            for n in ns:
                topn = top[0:n]
                hits = [j for j in topn if j in likes]
                recall[n].append(len(hits) / float(len(likes)))
            bar.update(item_id=str(numpy.mean(recall[max(ns)])))
        return [[numpy.mean(recall[n]), scipy.stats.sem(recall[n])] for n in ns]


    def precision(self, likes_dict, exclude_dict, n=100, n_users=None, exclude_items=None):
        precision = []
        try:
            if n_users is None:
                users = likes_dict.keys()
            else:
                numpy.random.seed(1)
                users = numpy.random.choice(likes_dict.keys(), replace=False, size=n_users)
            bar = pyprind.ProgBar(len(users))
            for inx, top in enumerate(self.topN(users, exclude_dict, n=n, exclude_items=exclude_items)):
                user = users[inx]
                likes = likes_dict[user]
                hits = [j for j in top if j in likes]
                precision.append(len(hits) / float(n))
                bar.update(item_id=str(numpy.mean(precision)))
        except Exception as e:
            sys.stderr.write(str(e))
        finally:
            return numpy.mean(precision), scipy.stats.sem(precision), precision

    def __repr__(self):
        repr_str = self.__class__.__name__
        for name, val in self.params():
            if isinstance(val, bool):
                if val:
                    val_str = "T"
                else:
                    val_str = "F"
            elif isinstance(val, (int, long, float, complex)):
                val_str = "%g" % val
            else:
                val_str = str(val)
            repr_str += " " + name + " " + val_str
        return repr_str


class BPRModel(Model):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, lambda_u=0.1, lambda_v=0.1, lambda_b=0.1,
                 use_bias=True, use_factors=True,
                 learning_rate=0.05, loss_f="sigmoid", margin=1, hard_case_margin=0, uneven_sample=False,
                 per_user_sample=10,
                 warp=False, batch_size=-1, bias_init=0.0,
                 bias_range=(-numpy.Infinity, numpy.Infinity)):

        Model.__init__(self, n_users, n_items)
        self.n_factors = n_factors
        self.use_factors = use_factors
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        # cache train_dict
        self.train_dict = None
        # cache sample_gen
        self.sample_generator = None
        self.valid_sample_generator = None

        self.lambda_b = lambda_b
        self.b = b
        self.learning_rate = learning_rate
        self.per_user_sample = per_user_sample
        self.batch_size = batch_size
        self.uneven_sample = uneven_sample
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.bias_range = bias_range
        self.warp = warp
        self.loss_f = loss_f
        self.margin = margin
        self.hard_case_margin = hard_case_margin
        if use_factors:
            if U is None:
                self.U = theano.shared(
                    value=numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users, n_factors)).astype(
                        theano.config.floatX) / 5,
                    name='U',
                    borrow=True
                )
            else:
                self.U = theano.shared(
                    value=U.astype(theano.config.floatX),
                    name='U',
                    borrow=True
                )

            if V is None:
                # randomly initialize user latent vectors
                self.V = theano.shared(
                    value=numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_items, n_factors)).astype(
                        theano.config.floatX) / 5,
                    name='V',
                    borrow=True
                )

            else:
                self.V = theano.shared(
                    value=V.astype(theano.config.floatX),
                    name='V',
                    borrow=True
                )
        else:
            self.U = theano.shared(value=numpy.zeros((self.n_users, 1)).astype(theano.config.floatX))
            self.V = theano.shared(value=numpy.zeros((self.n_items, 1)).astype(theano.config.floatX))

        if b is None:
            self.b = theano.shared(
                value=numpy.zeros((self.n_items, 1)).astype(theano.config.floatX) + bias_init,
                name='b',
                borrow=True
            )
        else:
            self.b = theano.shared(
                value=b.astype(theano.config.floatX),
                name='b',
                borrow=True
            )

        self.triplet = T.lmatrix('triplet')
        # user id
        self.i = self.triplet[:, 0]
        # a vector of pos item ids
        self.j_pos = self.triplet[:, 1]
        # a vector of neg item ids
        self.j_neg = self.triplet[:, 2]

        self.sample_weights = T.fvector()

        all_j = T.concatenate((self.j_pos, self.j_neg))
        one = numpy.asarray([1.0], dtype="float32").reshape((1, 1))

        # for those samples that do appear in the triplet, set the count to 0 first
        self.item_sample_counts = T.inc_subtensor(T.zeros((self.n_items, 1), theano.config.floatX)[all_j],
                                                  one)

        self.unique_j = T.arange(n_items)[self.item_sample_counts[:,0].nonzero()]

        self.user_sample_counts = T.inc_subtensor(T.zeros((self.n_users, 1), theano.config.floatX)[self.i], one)

        # if use adagrad or not
        self.adagrad = T.iscalar('adagrad')

        # learning rate
        # self.lr = T.fscalar('lr')

        # user vector
        self.u = self.U[self.i]

        # item vectors
        self.v_pos = self.V[self.j_pos]
        self.v_neg = self.V[self.j_neg]
        #  biases
        self.b_pos = self.b[self.j_pos]
        self.b_neg = self.b[self.j_neg]

        # these variables will be initialized later
        self.f = None
        self.global_f = None
        self.score_f = None
        self.validate_f = None
        self.hard_cases = dict()
        # user batch size for scoring (may be reduced to fit in GPU memory)
        self.n_user_count = 4
        self.value_f = None
        self.warp_f = None


    def params(self):
        return super(BPRModel, self).params() + [
            ["lr", self.learning_rate],
            ["use_f", self.use_factors],
            ["factors", self.n_factors],
            ["l_u", self.lambda_u],
            ["l_v", self.lambda_v],
            ["l_b", self.lambda_b],
            ["per_u", self.per_user_sample],
            ["bias", self.use_bias],
            ["uneven", self.uneven_sample]
        ]

    def l2_reg(self):
        ret = []
        if self.lambda_u != 0.0:
            ret.append((self.U, self.lambda_u))
        if self.lambda_v != 0.0:
            ret.append((self.V, self.lambda_v))
        if self.lambda_b != 0.0:
            ret.append((self.b - self.bias_init, self.lambda_b))
        return ret

    def l1_reg(self):
        return []

    def updates(self):
        updates = []
        if self.use_factors:
            updates += [[self.U, self.user_sample_counts],
                        # self.item_sample_counts_with_minus is Infinity for those items that DO NOT appear in the triplet
                        # the gradient for those items should be zeros, we do so to avoid 0/0 = NAN issue
                        [self.V, self.item_sample_counts],
                        ]

        if self.use_bias:
            def limit(b): return T.minimum(T.maximum(b, self.bias_range[0]), self.bias_range[1])

            updates += [[self.b, self.item_sample_counts, limit]]
        return updates

    def censor_updates(self):
        return []

    def delta(self):
        # user vector
        return self.factor_delta() + self.bias_delta()

    def factor_delta(self):
        # user vector
        delta = (self.U[self.i] * (self.V[self.j_pos] - self.V[self.j_neg])).sum(axis=1)
        return delta

    def bias_delta(self):
        if self.use_bias:
            return self.b[self.j_pos, 0] - self.b[self.j_neg, 0]
        return 0

    def scores_ij(self, i, j):
        scores = (self.U[i] * (self.V[j])).sum(axis=1)
        if self.use_bias:
            scores += self.b[j, 0]
        return scores

    def bias_score(self):
        if self.use_bias:
            return self.b.reshape((self.n_items,))
        else:
            return 0

    def factor_score(self):
        return T.dot(self.U[self.i], self.V.T)

    def score_fn(self):
        return self.factor_score() + self.bias_score()

    def scores_for_users(self, users):
        if self.score_f is None:
            self.score_f = theano.function(
                inputs=[self.i],
                outputs=self.score_fn()
            )
        while self.n_user_count > 2:
            try:
                for users in numpy.array_split(numpy.asarray(users, dtype="int32"),
                                               max(1, len(users) // self.n_user_count)):
                    for scores in self.score_f(users):
                        yield scores
                break
            except Exception as e:
                print e
                self.n_user_count /= 2
                continue

    def loss(self, delta):
        if self.loss_f == "sigmoid":
            return -T.log(T.nnet.sigmoid(delta))
        elif self.loss_f == "hinge":
            return T.maximum(0, self.margin - delta)

    def cost(self):
        # user vector

        delta = self.delta()
        losses = self.loss(delta)
        weighted_losses = losses * self.sample_weights
        # re-weight loss based on their approximate rank
        cost = weighted_losses.sum()
        return cost, losses

    def regularization_cost(self):
        cost = 0
        for term, l in self.l2_reg():
            if float(l) != float(0.0):
                cost += ((term ** 2) ).sum() * l
        for term, l in self.l1_reg():
            if float(l) != float(0.0):
                cost += (abs(term)).sum() * l
        return cost

    def warp_func(self):
        count = int(self.warp)
        index_of_same_samples = (T.arange(self.triplet.shape[0]/count) * count)
        pos_scores = T.repeat(self.scores_ij(self.i[index_of_same_samples], self.j_pos[index_of_same_samples]), count)
        neg_scores = self.scores_ij(self.i, self.j_neg)
        losses = T.maximum(0, self.margin - (pos_scores - neg_scores))
        losses = losses.reshape((losses.shape[0] / count, count))
        violations = (losses > 0).sum(1)
        weights = T.switch(violations > 0, T.cast(T.log(self.n_items * violations / count), "float32"), T.zeros((violations.shape[0],), dtype="float32"))
        active_sample_index = T.argmax(losses, axis=1) + (T.arange(self.triplet.shape[0]/count) * count)
        active_samples = self.triplet[active_sample_index]
        return theano.function([self.triplet], [active_samples, weights])

    def gen_updates(self, cost, per_sample_losses):
        update_list = []
        for update in self.updates():
            param = update[0]
            dividend = update[1]
            regularization_cost = self.regularization_cost()

            history = theano.shared(
                value=numpy.zeros(param.get_value().shape).astype(theano.config.floatX),
                borrow=True
            )
            gradient = T.grad(cost=cost, wrt=param, disconnected_inputs='ignore') / (T.cast(dividend, "float32") + 1E-10)
            gradient += T.grad(cost=regularization_cost, wrt=param, disconnected_inputs='ignore')
            new_history = ifelse(self.adagrad > 0, (history ) + (gradient ** float(2)), history)
            update_list += [[history, new_history]]
            adjusted_grad = ifelse(self.adagrad > 0, gradient / ((new_history ** float(0.5)) + float(1e-10)), gradient)
            new_param = param - ((adjusted_grad ) * float(self.learning_rate))
            if len(update) == 3:
                new_param = update[2](new_param)
            # censored updates
            for censored_param, limit in self.censor_updates():
                if param == censored_param:
                    col_norms = T.sqrt(T.sum(T.sqr(new_param), axis=1))
                    desired_norms = T.clip(col_norms, 0, limit)
                    new_param *= (desired_norms / (1e-10 + col_norms)).reshape((new_param.shape[0], 1))
            update_list += [[param, new_param]]
        return update_list

    def sample(self, train_dict, exclude_items):
        return create_sampler(train_dict, None, self.n_items,
                                               exclude_items=exclude_items,
                                               per_user_sample=self.per_user_sample,
                                               batch_size=self.batch_size,
                                               uneven=self.uneven_sample,
                                               warp=self.warp)

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False, profile=False, exclude_items=None):
        if self.warp is not False and self.warp_f is None:
            self.warp_f = self.warp_func()
        if self.f is None:
            self.check_signature(train_dict)
            self.train_dict = train_dict
            self.sample_generator = self.sample(train_dict, exclude_items)


            cost, per_sample_losses = self.cost()

            self.f = theano.function(
                inputs=[self.triplet, self.sample_weights, self.adagrad],
                outputs=[cost, per_sample_losses],
                updates=self.gen_updates(cost, per_sample_losses),
                profile=profile,
                on_unused_input='warn'
            )

        # perform latent vector updates #
        epoch_index = 0
        losses = []
        sample_time = 0
        training_time = 0
        sample_start = time.time()
        if adagrad:
            adagrad_val = 1
        else:
            adagrad_val = 0

        for triplet in self.sample_generator:
            sample_time += time.time() - sample_start
            training_start = time.time()
            if self.warp is not False:
                triplet, weights = self.warp_f(triplet)
            else:
                weights = 1.0
            # update latent vectors
            loss, per_sample_losses = self.f(triplet, weights, adagrad_val)
            losses.append(loss)
            #print numpy.sum(per_sample_losses > 0) / float(len(per_sample_losses))
            training_time += time.time() - training_start
            epoch_index += 1
            if epoch_index == epoch:
                sys.stderr.write(
                    "Train Time: %g Sample Time: %g\n" % (training_time, sample_time))
                sys.stderr.flush()
                return numpy.mean(losses)
            sample_start = time.time()




    def validate(self, train_dict, valid_dict, per_user_sample=100):
        if self.validate_f is None:
            # compute AUC
            delta = self.delta()
            self.validate_f = theano.function(
                inputs=[self.triplet],
                outputs=T.switch(T.gt(delta, 0), 1.0, 0.0)
            )
            self.valid_sample_generator = self.sample(train_dict, None)
            self.train_valid_sample_generator = self.sample(valid_dict)

        results = None
        if train_dict is None:
            gen = self.train_valid_sample_generator
        else:
            gen = self.valid_sample_generator
        # sample triplet with the sample generator
        for triplet in gen:
            # update latent vectors
            results = self.validate_f(triplet)
            break
        results = results.reshape((len(results) / per_user_sample, per_user_sample))
        aucs = numpy.mean(results, axis=1)
        return numpy.mean(aucs), scipy.stats.sem(aucs)

    def __getstate__(self):
        import copy
        ret = copy.copy(self.__dict__)
        ret["f"] = None
        ret["validate_f"] = None
        ret["score_f"] = None
        ret["sample_generator"] = None
        ret["train_valid_sample_generator"] = None
        ret["valid_sample_generator"] = None
        ret["train_dict"] = None
        ret["valid_dict"] = None
        ret["trials"] = None
        ret["hard_cases"] = dict()
        ret["warp_f"] = None
        return ret


class BPRModelWithVisualBias(BPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U=None, V=None, b=None, visual_b=None,
                 use_bias=True, use_factors=True, use_visual_bias=True,
                 lambda_u=0.1, lambda_v=0.1, lambda_b=0.1, lambda_visual_b=0.1, learning_rate=0.01,
                 per_user_sample=10):
        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_b, use_bias,
                          use_factors,
                          learning_rate,
                          per_user_sample)
        self.lambda_visual_b = lambda_visual_b
        self.use_visual_bias = use_visual_bias
        self.V_features = theano.shared(
            value=V_features.astype(theano.config.floatX),
            name='V_visual',
            borrow=True
        )

        raw_visual_feature_factors = len(self.V_features.get_value()[0])

        # create visual bias
        if self.use_visual_bias:
            if visual_b is None:
                self.visual_b = theano.shared(
                    value=numpy.random.normal(0, 0.01, raw_visual_feature_factors).astype(theano.config.floatX),
                    name='visual_b',
                    borrow=True
                )
            else:
                self.visual_b = theano.shared(
                    value=visual_b.astype(theano.config.floatX),
                    name='visual_b',
                    borrow=True
                )
        else:
            self.visual_b = theano.shared(
                value=numpy.zeros(raw_visual_feature_factors).astype(theano.config.floatX),
                name='visual_b',
                borrow=True
            )

        self.V_bias = T.dot(self.V_features, self.visual_b)

    def params(self):
        return super(BPRModelWithVisualBias, self).params() + [
            ["vbias", self.use_visual_bias],
            ["l_vb", self.lambda_visual_b]
        ]

    def copy(self):
        z = super(BPRModelWithVisualBias, self).copy()
        z.update(
            {"visual_b": self.visual_b.get_value(),
             "use_visual_bias": self.use_visual_bias,
             "lambda_visual_b": self.lambda_visual_b
             }
        )
        return z

    def bias_delta(self):
        delta = super(BPRModelWithVisualBias, self).bias_delta()
        delta += self.V_bias[self.j_pos] - self.V_bias[self.j_neg]
        return delta

    def score_fn(self):
        return super(BPRModelWithVisualBias, self).score_fn() + self.V_bias

    def l2_reg(self):
        return BPRModel.l2_reg(self) + \
               [[self.visual_b, self.lambda_visual_b]]

    def l1_reg(self):
        return BPRModel.l1_reg(self) + \
               [[self.visual_b, self.lambda_visual_b]]

    def updates(self):
        updates = super(BPRModelWithVisualBias, self).updates()
        if self.use_visual_bias:
            updates += [[self.visual_b, self.n_users * self.per_user_sample]]
        return updates


class VisualBPRAbstractModel(BPRModelWithVisualBias):
    def __init__(self, n_factors, n_users, n_items, n_embedding_dim, V_features, U=None, V=None, U_features=None,
                 b=None, visual_b=None,
                 weights=None, u_weights=None, use_bias=True, use_factors=True, use_visual_bias=True,
                 use_visual_offset=False,
                 nonlinear=True, activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 learning_rate=0.01, per_user_sample=10):
        BPRModelWithVisualBias.__init__(self, n_factors, n_users, n_items, V_features, U, V, b, visual_b,
                                        use_bias, use_factors, use_visual_bias, lambda_u, lambda_v, lambda_b,
                                        lambda_visual_b,
                                        learning_rate,
                                        per_user_sample)

        self.lambda_weight_l1 = lambda_weight_l1
        self.lambda_weight_l2 = lambda_weight_l2
        self.n_embedding_dim = n_embedding_dim
        self.use_visual_offset = use_visual_offset
        self.lambda_v_offset = lambda_v_offset
        self.n_layers = n_layers
        # create weights
        raw_visual_feature_factors = len(self.V_features.get_value()[0])
        self.nonlinear = nonlinear
        if self.nonlinear:
            self.activation = activation
        else:
            self.activation = None

        if weights is not None:
            self.sample_weights = [theano.shared(numpy.asarray(
                w,
                dtype=theano.config.floatX
            ), borrow=True)
                                   for w in weights
                                   ]
        else:
            if self.nonlinear:
                self.sample_weights = [
                    theano.shared(numpy.asarray(
                        numpy.random.uniform(
                            low=-numpy.sqrt(6. / (raw_visual_feature_factors + 128)),
                            high=numpy.sqrt(6. / (raw_visual_feature_factors + 128)),
                            size=(raw_visual_feature_factors, 128)
                        ),
                        dtype=theano.config.floatX
                    ), borrow=True, name='weights_' + str(0)),
                    theano.shared(numpy.asarray(
                        numpy.zeros((128,), dtype=theano.config.floatX),
                        dtype=theano.config.floatX
                    ), borrow=True, name='weight_bias_' + str(0))
                ]

                for i in range(1, self.n_layers - 1):
                    self.sample_weights += [
                        theano.shared(numpy.asarray(
                            numpy.random.uniform(
                                low=-numpy.sqrt(6. / (128 + 128)),
                                high=numpy.sqrt(6. / (128 + 128)),
                                size=(128, 128)
                            ),
                            dtype=theano.config.floatX
                        ), borrow=True, name='weights_' + str(i)),
                        theano.shared(numpy.asarray(
                            numpy.zeros((128,), dtype=theano.config.floatX),
                            dtype=theano.config.floatX
                        ), borrow=True, name='weight_bias_' + str(i))
                    ]

                self.sample_weights += [theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (128 + n_embedding_dim)),
                        high=numpy.sqrt(6. / (128 + n_embedding_dim)),
                        size=(128, n_embedding_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True, name='weights_end')]
            # linear embedding
            else:
                self.sample_weights = [theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                        high=numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                        size=(raw_visual_feature_factors, n_embedding_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True, name='weights')]

        self.V_embedding = self.embedding(self.V_features, self.sample_weights)

        if use_visual_offset:
            self.V_offset = theano.shared(
                value=numpy.random.normal(0, 1 / (self.n_embedding_dim ** 0.5), (n_items, n_embedding_dim)).astype(
                    theano.config.floatX),
                name='V_offset',
                borrow=True
            )
            self.V_embedding += self.V_offset
        else:
            self.V_offset = None

        ## User features ##
        if U_features is not None:

            self.U_features = theano.shared(
                value=U_features.astype(theano.config.floatX),
                name='U_features',
                borrow=True
            )
            self.use_user_feature = True
            raw_user_feature_factors = len(self.U_features.get_value()[0])
            if self.nonlinear:
                if u_weights is None:
                    self.u_weights = [
                        theano.shared(numpy.asarray(
                            numpy.random.uniform(
                                low=-numpy.sqrt(6. / (raw_user_feature_factors + 128)),
                                high=numpy.sqrt(6. / (raw_user_feature_factors + 128)),
                                size=(raw_user_feature_factors, 128)
                            ),
                            dtype=theano.config.floatX
                        ), borrow=True, name='u_weights_1'),
                        theano.shared(numpy.asarray(
                            numpy.random.uniform(
                                low=-numpy.sqrt(6. / (128 + n_embedding_dim)),
                                high=numpy.sqrt(6. / (128 + n_embedding_dim)),
                                size=(128, n_embedding_dim)
                            ),
                            dtype=theano.config.floatX
                        ), borrow=True, name='u_weights_2'),
                        theano.shared(numpy.asarray(
                            numpy.zeros((128,), dtype=theano.config.floatX),
                            dtype=theano.config.floatX
                        ), borrow=True, name='u_weight_bias')
                    ]
                else:
                    self.u_weights = [theano.shared(numpy.asarray(
                        u_weights[0],
                        dtype=theano.config.floatX
                    ), borrow=True, name='u_weights_1'),
                        theano.shared(numpy.asarray(
                            u_weights[1],
                            dtype=theano.config.floatX
                        ), borrow=True, name='u_weights_2'),
                        theano.shared(numpy.asarray(
                            u_weights[2],
                            dtype=theano.config.floatX
                        ), borrow=True, name='u_weight_bias')
                    ]
            else:
                if u_weights is None:
                    self.u_weights = [theano.shared(numpy.asarray(
                        numpy.random.uniform(
                            low=-numpy.sqrt(6. / (raw_user_feature_factors + n_embedding_dim)),
                            high=numpy.sqrt(6. / (raw_user_feature_factors + n_embedding_dim)),
                            size=(raw_user_feature_factors, n_embedding_dim)
                        ),
                        dtype=theano.config.floatX
                    ), borrow=True, name='u_weights')]
                else:
                    self.u_weights = [theano.shared(numpy.asarray(
                        u_weights[0],
                        dtype=theano.config.floatX
                    ), borrow=True, name='u_weights')]
            self.U_embedding = self.embedding(self.U_features, self.u_weights)
        else:
            self.u_weights = []
            self.use_user_feature = False

    def embedding(self, features, weights):
        if self.nonlinear:
            activation_fn = eval(self.activation)
            ret = features
            for i in range(self.n_layers - 1):
                ret = activation_fn(T.dot(ret, weights[i * 2]) + weights[i * 2 + 1])
            return T.dot(ret, weights[-1])
        else:
            return T.dot(features, weights[0])

    def item_cold_start(self, test_dict, item_features, n_closest=500):
        def to_T(m):
            return theano.shared(
                value=m.astype(theano.config.floatX),
                borrow=True
            )

        V_embedding = theano.function([], self.V_embedding)()
        V = self.V.get_value()

        def n_closest_V(v, n):
            if n_closest > 0:
                return numpy.sum(V[numpy.argpartition(-numpy.inner(v, V_embedding), n)[:n]], axis=0)
            else:
                return numpy.zeros(V.shape[1]) + 1

        tmp = [self.V, self.V_offset, self.V_embedding, self.V_bias]
        sys.stderr.write("Zeroing parameters\n")
        try:
            n_items = len(item_features)

            self.V_embedding = self.embedding(to_T(item_features), self.sample_weights)
            cold_V_embedding = theano.function([], self.V_embedding)()

            if self.use_factors:
                self.V = to_T(numpy.asarray([n_closest_V(v, n_closest) for v in cold_V_embedding]))

            if self.use_visual_offset:
                self.V_offset = to_T(numpy.zeros((n_items, self.n_embedding_dim)))
            if self.use_bias:
                self.V_bias = to_T(numpy.zeros(n_items))
            return self.auc(test_dict, None)
        except Exception as e:
            sys.stderr.write("Error" + str(e) + "\n")

        finally:
            sys.stderr.write("Recover parameters\n")
            self.V, self.V_offset, self.V_embedding, self.V_bias = tmp

    def copy(self):
        z = super(VisualBPRAbstractModel, self).copy()
        V_offset = None
        if self.V_offset is not None:
            V_offset = self.V_offset.get_value()
        z.update(
            {"lambda_weight_l1": self.lambda_weight_l1,
             "lambda_weight_l2": self.lambda_weight_l2,
             "lambda_v_offset": self.lambda_v_offset,
             "weights": [w.get_value() for w in self.sample_weights],
             "u_weights": [w.get_value() for w in self.u_weights],
             "V_offset": V_offset,
             "use_visual_offset": self.use_visual_offset,
             "use_user_features": self.use_user_feature,
             "n_embedding_dim": self.n_embedding_dim,
             "nonlinear": self.nonlinear,
             "n_layers": self.n_layers}
        )
        return z

    def params(self):
        return super(VisualBPRAbstractModel, self).params() + [
            ["l_w_1", self.lambda_weight_l1],
            ["l_w_2", self.lambda_weight_l2],
            ["l_voff", self.lambda_v_offset],
            ["voff", self.use_visual_offset],
            ["u_fea", self.use_user_feature],
            ["edim", self.n_embedding_dim],
            ["nonlinear", self.nonlinear],
            ["layers", self.n_layers],
            ["act", self.activation]]

    def l2_reg(self):
        reg = super(VisualBPRAbstractModel, self).l2_reg() + [[w, self.lambda_weight_l2] for w in self.sample_weights]
        if self.use_visual_offset:
            reg += [[self.V_offset, self.lambda_v_offset]]
        return reg

    def l1_reg(self):
        return super(VisualBPRAbstractModel, self).l1_reg() + [[w, self.lambda_weight_l1] for w in self.sample_weights]

    def updates(self):
        updates = super(VisualBPRAbstractModel, self).updates()

        updates += [
            [w, self.n_users * self.per_user_sample]
            for w in self.sample_weights + self.u_weights]

        if self.use_visual_offset:
            updates += [
                [self.V_offset, self.item_sample_counts.reshape((self.n_items, 1))]]
        return updates


class VisualBPRConcatModel(VisualBPRAbstractModel):
    def __init__(self, n_factors, n_visual_factors, n_users, n_items, V_features, U=None, V=None, U_features=None,
                 U_visual=None, VU_features=None, b=None,
                 visual_b=None,
                 weights=None, u_weights=None, use_bias=True, use_factors=True, use_visual_bias=True,
                 use_visual_offset=False, nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01, learning_rate=0.01,
                 per_user_sample=10, fine_tune=False):

        super(VisualBPRConcatModel, self).__init__(n_factors, n_users, n_items, n_visual_factors, V_features, U=U, V=V,
                                                   b=b,
                                                   U_features=U_features,
                                                   visual_b=visual_b, weights=weights, u_weights=u_weights,
                                                   use_bias=use_bias,
                                                   use_factors=use_factors,
                                                   use_visual_bias=use_visual_bias, use_visual_offset=use_visual_offset,
                                                   nonlinear=nonlinear, activation=activation,
                                                   n_layers=n_layers,
                                                   lambda_u=lambda_u, lambda_v=lambda_v,
                                                   lambda_v_offset=lambda_v_offset,
                                                   lambda_b=lambda_b,
                                                   lambda_visual_b=lambda_visual_b, lambda_weight_l2=lambda_weight_l2,
                                                   lambda_weight_l1=lambda_weight_l1,
                                                   learning_rate=learning_rate,
                                                   per_user_sample=per_user_sample)

        self.n_visual_factors = n_visual_factors
        self.U_visual = U_visual
        # whether to fix U, V, U_visual and only adjust weights
        self.fine_tune = fine_tune
        # create user visual factor matrix
        if U_visual is None:
            self.U_visual = theano.shared(
                value=numpy.random.normal(0, 1 / (self.n_visual_factors ** 0.5), (n_users, n_visual_factors)).astype(
                    theano.config.floatX),
                name='U_visual',
                borrow=True
            )
        else:
            self.U_visual = theano.shared(
                value=U_visual.astype(theano.config.floatX),
                name='U_visual',
                borrow=True
            )
        if self.use_user_feature:
            if VU_features is None:
                self.VU_features = theano.shared(
                    value=numpy.random.normal(0, 1 / (self.n_visual_factors ** 0.5),
                                              (n_items, n_visual_factors)).astype(theano.config.floatX),
                    name='VU_features',
                    borrow=True
                )
            else:

                self.VU_features = theano.shared(
                    value=VU_features.astype(theano.config.floatX),
                    name='VU_features',
                    borrow=True
                )
        else:
            self.VU_features = None

    def copy(self):
        z = super(VisualBPRConcatModel, self).copy()
        z.update(
            {"U_visual": self.U_visual.get_value(),
             }
        )
        if self.use_user_feature:
            z.update(
                {"VU_features": self.VU_features.get_value()
                 }
            )
        return z

    def updates(self):

        updates = super(VisualBPRConcatModel, self).updates()

        updates += [
            [self.U_visual, self.per_user_sample]
        ]
        if self.use_user_feature:
            updates += [[self.VU_features, self.item_sample_counts.reshape((self.n_items, 1))]]
        return updates

    def full_V(self):
        V = [self.V_embedding]
        if self.use_factors:
            V.append(self.V)
        if self.use_user_feature:
            V.append(self.VU_features)
        return T.concatenate(V, axis=1)

    def full_U(self):
        U = [self.U_visual]
        if self.use_factors:
            U.append(self.U)
        if self.use_user_feature:
            U.append(self.U_embedding)
        return T.concatenate(U, axis=1)

    def factor_delta(self):
        # compute difference of visual features in the embedding
        delta = super(VisualBPRConcatModel, self).delta()
        delta += (self.U_visual[self.i] * (self.V_embedding[self.j_pos] - self.V_embedding[self.j_neg])).sum(axis=1)
        if self.use_user_feature:
            delta += (self.U_embedding[self.i] * (self.VU_features[self.j_pos] - self.VU_features[self.j_neg])).sum(
                axis=1)
        return delta

    def score_fn(self):
        fn = super(VisualBPRConcatModel, self).score_fn() + T.dot(self.U_visual[self.i], self.V_embedding.T)
        if self.use_user_feature:
            fn += T.dot(self.U_embedding[self.i], self.VU_features.T)
        return fn

    def l2_reg(self):
        reg = super(VisualBPRConcatModel, self).l2_reg() + [[self.U_visual, self.lambda_u]]
        if self.use_user_feature:
            reg += [[self.VU_features, self.lambda_v]]
        return reg


class VisualBPRConcatNormModel(VisualBPRConcatModel):
    def __init__(self, n_factors, n_visual_factors, n_users, n_items, V_features, U=None, V=None, U_features=None,
                 U_visual=None, b=None,
                 visual_b=None,
                 weights=None, u_weights=None, use_bias=True, use_factors=True, use_visual_bias=True,
                 use_visual_offset=False, nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01, learning_rate=0.01,
                 per_user_sample=10):
        super(VisualBPRConcatNormModel, self).__init__(n_factors, n_visual_factors, n_users, n_items, V_features, U=U,
                                                       V=V, U_features=U_features,
                                                       U_visual=U_visual, b=b,
                                                       visual_b=visual_b, weights=weights, u_weights=u_weights,
                                                       use_bias=use_bias,
                                                       use_factors=use_factors,
                                                       use_visual_bias=use_visual_bias,
                                                       use_visual_offset=use_visual_offset, nonlinear=nonlinear,
                                                       activation=activation,
                                                       n_layers=n_layers,
                                                       lambda_u=lambda_u, lambda_v=lambda_v,
                                                       lambda_v_offset=lambda_v_offset,
                                                       lambda_b=lambda_b,
                                                       lambda_visual_b=lambda_visual_b,
                                                       lambda_weight_l2=lambda_weight_l2,
                                                       lambda_weight_l1=lambda_weight_l1,
                                                       learning_rate=learning_rate,
                                                       per_user_sample=per_user_sample)

    def normalize(self, variable):
        return variable / ((T.sum(variable ** 2, axis=1) ** 0.5).reshape((variable.shape[0], 1)))

    def full_V(self):
        V = [self.normalize(self.V_embedding)]
        if self.use_factors:
            V.append(self.normalize(self.V))
        if self.use_user_feature:
            V.append(self.normalize(self.VU_features))
        return self.normalize(T.concatenate(V, axis=1))

    def full_U(self):
        U = [self.normalize(self.U_visual)]
        if self.use_factors:
            U.append(self.normalize(self.U))
        if self.use_user_feature:
            U.append(self.normalize(self.U_embedding))
        return self.normalize(T.concatenate(U, axis=1))

    def factor_delta(self):
        U = self.full_U()
        V = self.full_V()
        return (U[self.i] * (V[self.j_pos] - V[self.j_neg])).sum(axis=1) + \
               self.b[self.j_pos] - self.b[self.j_neg] + self.V_bias[self.j_pos] - self.V_bias[self.j_neg]

    def score_fn(self):
        U = self.full_U()
        V = self.full_V()
        return T.dot(U[self.i], V.T) + \
               self.b + self.V_bias


class VisualBPRConcatNormContrastiveModel(VisualBPRConcatNormModel):
    def __init__(self, n_factors, n_visual_factors, n_users, n_items, V_features, U=None, V=None, U_features=None,
                 U_visual=None, b=None,
                 visual_b=None,
                 weights=None, u_weights=None, use_bias=True, use_factors=True, use_visual_bias=True,
                 use_visual_offset=False, nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01, learning_rate=0.01,
                 per_user_sample=10, margin=0.2):
        super(VisualBPRConcatNormContrastiveModel, self).__init__(
            n_factors, n_visual_factors, n_users, n_items, V_features,
            U=U, V=V, U_features=U_features,
            U_visual=U_visual, b=b,
            visual_b=visual_b, weights=weights, u_weights=u_weights,
            use_bias=use_bias,
            use_factors=use_factors,
            use_visual_bias=use_visual_bias,
            use_visual_offset=use_visual_offset, nonlinear=nonlinear,
            activation=activation,
            n_layers=n_layers,
            lambda_u=lambda_u, lambda_v=lambda_v,
            lambda_v_offset=lambda_v_offset,
            lambda_b=lambda_b,
            lambda_visual_b=lambda_visual_b,
            lambda_weight_l2=lambda_weight_l2,
            lambda_weight_l1=lambda_weight_l1,
            learning_rate=learning_rate,
            per_user_sample=per_user_sample)
        self.margin = margin

    def copy(self):
        z = super(VisualBPRConcatNormContrastiveModel, self).copy()
        z.update(
            {"margin": self.margin}
        )
        return z

    def params(self):
        return super(VisualBPRConcatNormContrastiveModel, self).params() + [
            ["margin", self.margin]]

    def loss(self, delta):
        return T.maximum(0, self.margin - delta)


class VisualBPRConcatNormContrastiveModelWeighted(VisualBPRConcatNormContrastiveModel):
    def __init__(self, n_factors, n_visual_factors, n_users, n_items, V_features, U=None, V=None, U_features=None,
                 U_visual=None, b=None,
                 visual_b=None,
                 weights=None, u_weights=None, global_ws=None, U_ws=None, use_bias=True, use_factors=True,
                 use_visual_bias=True, use_visual_offset=False, nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 lambda_U_weights=10,
                 learning_rate=0.01,
                 per_user_sample=10, margin=0.2):
        super(VisualBPRConcatNormContrastiveModelWeighted, self).__init__(
            n_factors, n_visual_factors, n_users, n_items, V_features,
            U=U, V=V, U_features=U_features,
            U_visual=U_visual, b=b,
            visual_b=visual_b, weights=weights, u_weights=u_weights,
            use_bias=use_bias,
            use_factors=use_factors,
            use_visual_bias=use_visual_bias,
            use_visual_offset=use_visual_offset, nonlinear=nonlinear,
            activation=activation,
            n_layers=n_layers,
            lambda_u=lambda_u, lambda_v=lambda_v,
            lambda_v_offset=lambda_v_offset,
            lambda_b=lambda_b,
            lambda_visual_b=lambda_visual_b,
            lambda_weight_l2=lambda_weight_l2,
            lambda_weight_l1=lambda_weight_l1,
            learning_rate=learning_rate,
            per_user_sample=per_user_sample)
        self.margin = margin
        self.lambda_U_weights = lambda_U_weights
        if global_ws is None:
            self.global_ws = theano.shared(
                value=numpy.asarray([1, 1, 1]).astype(theano.config.floatX),
                name='global_ws',
                borrow=True
            )
        else:
            self.global_ws = theano.shared(
                value=global_ws.astype(theano.config.floatX),
                name='global_ws',
                borrow=True
            )
        if U_ws is None:
            self.U_ws = theano.shared(
                value=numpy.zeros((self.n_users, 3)).astype(theano.config.floatX),
                name='ws',
                borrow=True
            )
        else:
            self.U_ws = theano.shared(
                value=U_ws.astype(theano.config.floatX),
                name='ws',
                borrow=True
            )

    def full_V(self):
        V = [self.normalize(self.V_embedding)]
        if self.use_factors:
            V.append(self.normalize(self.V))
        if self.use_user_feature:
            V.append(self.normalize(self.VU_features))
        return self.normalize(T.concatenate(V, axis=1))

    def full_U(self):
        shape = (self.U_ws.shape[0], 1)
        U = [self.normalize(self.U_visual) * T.maximum(0, (self.U_ws[:, 0] + self.global_ws[0])).reshape(shape)]
        if self.use_factors:
            U.append(self.normalize(self.U) * T.maximum(0, (self.U_ws[:, 1] + self.global_ws[1])).reshape(shape))
        if self.use_user_feature:
            U.append(
                self.normalize(self.U_embedding) * T.maximum(0, (self.U_ws[:, 2] + self.global_ws[2])).reshape(shape))
        return self.normalize(T.concatenate(U, axis=1))

    def updates(self):

        updates = super(VisualBPRConcatNormContrastiveModelWeighted, self).updates()
        updates += [[self.global_ws, self.n_users * self.per_user_sample],
                    [self.U_ws, self.per_user_sample]]
        return updates

    def l2_reg(self):
        return super(VisualBPRConcatNormContrastiveModelWeighted, self).l2_reg() + [[self.U_ws, self.lambda_U_weights]]

    def params(self):
        return super(VisualBPRConcatNormContrastiveModelWeighted, self).params() + [
            ["l_Uw", self.lambda_U_weights]]

    def copy(self):
        z = super(VisualBPRConcatNormContrastiveModelWeighted, self).copy()
        z.update(
            {"ws": self.global_ws.get_value(),
             "U_ws": self.U_ws.get_value(),
             "lambda_U_weights": self.lambda_U_weights}
        )
        return z


class VisualBPRStackModel(VisualBPRAbstractModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U=None, V=None, b=None, visual_b=None,
                 weights=None, use_bias=True, use_visual_bias=True, use_visual_offset=False, nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_u=0.1, lambda_v=0.1, lambda_v_offset=10, lambda_b=0.1, lambda_visual_b=0.1,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 learning_rate=0.01, per_user_sample=10):
        VisualBPRAbstractModel.__init__(self, n_factors, n_users, n_items, n_factors, V_features, U, V, b,
                                        visual_b, weights=weights, use_bias=use_bias, use_visual_bias=use_visual_bias,
                                        use_visual_offset=use_visual_offset, nonlinear=nonlinear,
                                        activation=activation, n_layers=n_layers,
                                        lambda_u=lambda_u, lambda_v=lambda_v, lambda_v_offset=lambda_v_offset,
                                        lambda_b=lambda_b,
                                        lambda_visual_b=lambda_visual_b, lambda_weight_l2=lambda_weight_l2,
                                        lambda_weight_l1=lambda_weight_l1,
                                        learning_rate=learning_rate,
                                        per_user_sample=per_user_sample)

    def factor_delta(self):
        # compute difference of visual features in the embedding
        delta = super(VisualBPRStackModel, self).delta()
        delta += (self.U[self.i] * (self.V_embedding[self.j_pos] - self.V_embedding[self.j_neg])).sum(axis=1)
        return delta

    def score_fn(self):
        return super(VisualBPRStackModel, self).score_fn() + T.dot(self.U[self.i], self.V_embedding.T)


class UserVisualModel(VisualBPRConcatModel):
    def __init__(self, n_factors, n_users, n_items, V_features, K=2, margin=1, U_visual=None, weights=None,
                 nonlinear=True,
                 activation="T.tanh", n_layers=2,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 lambda_u=0.1, lambda_mean_distance=10, learning_rate=0.01,
                 per_user_sample=10):
        self.K = K
        self.margin = margin
        self.mixture_weight_f = None
        self.lambda_mean_distance = lambda_mean_distance
        super(UserVisualModel, self).__init__(0, n_factors, n_users, n_items, V_features, U_visual=U_visual,
                                              weights=weights, use_bias=False, use_visual_bias=False,
                                              use_factors=False,
                                              use_visual_offset=False,
                                              nonlinear=nonlinear,
                                              activation=activation, n_layers=n_layers,
                                              lambda_u=lambda_u, lambda_v=0, lambda_b=0,
                                              lambda_visual_b=0, lambda_weight_l2=lambda_weight_l2,
                                              lambda_weight_l1=lambda_weight_l1,
                                              learning_rate=learning_rate,
                                              per_user_sample=per_user_sample)
        self.U_visual = theano.shared(
            value=numpy.random.normal(0, 0.1, (n_users * K, n_factors)).astype(theano.config.floatX),
            name='U_visual',
            borrow=True
        )
        self.mixture_weight = None
        self.U_mixture_portion = theano.shared(
            value=numpy.zeros((K - 1, n_users)).astype(theano.config.floatX) + (1.0 / K),
            name='U_mixture_portion',
            borrow=True
        )
        self.mixture_variance = theano.shared(
            value=numpy.zeros((K, n_users)).astype(theano.config.floatX) + 1.0,
            name='variance',
            borrow=True
        )
        self.U_mixture_portion_derived = T.log(T.concatenate((T.maximum(0, (
            1 - T.maximum(0, self.U_mixture_portion).sum(axis=0))).reshape((1, n_users)),
                                                              T.maximum(0, self.U_mixture_portion)), axis=0) + 0.001)
        self.U_mixture_portion_derived_long = self.U_mixture_portion_derived.reshape((self.K * self.n_users, 1))
        self.V_embedding_norm = self.normalize(self.V_embedding)
        self.U_visual_norm = self.normalize(self.U_visual)
        self.U_visual_reshape_norm = self.U_visual_norm.reshape((self.K, self.n_users, self.n_visual_factors))
        self.mixture_variance_long = self.mixture_variance.reshape((self.K * self.n_users, 1))

    def normalize(self, variable):
        return variable / ((T.sum(variable ** 2, axis=1) ** 0.5).reshape((variable.shape[0], 1)))

    def l2_reg(self):
        return super(UserVisualModel, self).l2_reg() + [[self.mixture_variance - 1, 1]]

    def updates(self):
        updates = super(UserVisualModel, self).updates()
        updates += [[self.U_mixture_portion, self.per_user_sample]]

        # updates += [[self.mixture_variance, 1]]
        return updates

    def cost(self):
        ret = super(UserVisualModel, self).cost()
        return ret

    def assign_cluster(self, i, j):
        # mixture_weight_reshape = self.mixture_weight.reshape((self.K, self.n_users))
        distance = ((self.U_visual_reshape_norm[:, i, :] - self.V_embedding_norm[j]) ** 2).sum(axis=2)
        return (-distance / (2 * (self.mixture_variance[:, i] ** 2)) - T.log(
            self.mixture_variance[:, i]) + self.U_mixture_portion_derived[:, i]).argmax(axis=0) * self.n_users + i

    def factor_delta(self):
        pos_j = self.assign_cluster(self.i, self.j_pos)
        pos_distance = ((self.U_visual_norm[pos_j, :] - self.V_embedding_norm[self.j_pos]) ** 2).sum(axis=1)
        pos_distance_p = -pos_distance / (2 * (self.mixture_variance_long[pos_j, 0] ** 2)) - T.log(
            self.mixture_variance_long[pos_j, 0]) + self.U_mixture_portion_derived_long[pos_j, 0]

        neg_j = self.assign_cluster(self.i, self.j_neg)
        neg_distance = ((self.U_visual_norm[neg_j, :] - self.V_embedding_norm[self.j_neg]) ** 2).sum(axis=1)
        neg_distance_p = -neg_distance / (2 * (self.mixture_variance_long[neg_j, 0] ** 2)) - T.log(
            self.mixture_variance_long[neg_j, 0]) + self.U_mixture_portion_derived_long[neg_j, 0]

        return pos_distance_p - neg_distance_p

    def create_mixture_weights(self, train):
        i_index = []
        j_index = []
        portion = []
        init = []
        for u, items in train.items():
            for item in items:
                i_index.append(u)
                j_index.append(item)
                portion.append(1.0 / (1 + float(len(items))))
        for i in range(self.n_users):
            item_count = 0
            if i in train:
                item_count = len(train[i])
            init.append(1.0 / float(self.K) / (1 + item_count))
        init *= self.K
        portion = numpy.asarray(portion, theano.config.floatX).reshape((len(portion), 1))
        init = numpy.asarray(init, theano.config.floatX).reshape((len(init), 1))
        mixture_weight = T.zeros((self.K * self.n_users, 1), theano.config.floatX) + init
        mixture_weight = T.inc_subtensor(mixture_weight[self.assign_cluster(i_index, j_index)], portion)
        return T.log(mixture_weight.reshape((self.K * self.n_users, 1)))

    def loss(self, delta):
        return T.maximum(0, self.margin - delta)

    def params(self):
        return super(UserVisualModel, self).params() + [
            ["margin", self.margin],
            ["K", self.K]
        ]

    def validate(self, train_dict, valid_dict, per_user_sample=100):
        if self.validate_f is None:
            # compute AUC
            delta = self.delta()
            self.validate_f = theano.function(
                inputs=[self.i, self.j_pos, self.j_neg],
                outputs=T.switch(T.gt(delta, 0), 1.0, 0.0)
            )
            self.valid_sample_generator = self.sample(valid_dict, train_dict, per_user_sample=per_user_sample)
        results = None
        # sample triplet with the sample generator
        for us, pos, neg in self.valid_sample_generator:
            # update latent vectors
            results = self.validate_f(us, pos, neg)
            break
        results = results.reshape((len(results) / per_user_sample, per_user_sample))
        aucs = numpy.mean(results, axis=1)
        return numpy.mean(aucs), scipy.stats.sem(aucs)


class KBPRModel(BPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=1, margin=1, hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=True,
                 warp=True,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_mean_distance=0.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 lambda_cov=10,
                 variance_mu=1.0,
                 bias_init=0.5,
                 update_mu=True,
                 update_density=True,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 per_user_sample=10,
                 normalization=False,
                 bias_range=(1E-6, 10),
                 batch_size=-1,
                 max_norm=1):
        if U is None:
            U = numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users * K, n_factors)).astype(
                theano.config.floatX) / 5

        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_bias,
                          use_bias=use_bias,
                          use_factors=True,
                          loss_f="hinge",
                          margin=margin,
                          learning_rate=learning_rate,
                          uneven_sample=uneven_sample,
                          per_user_sample=per_user_sample,
                          batch_size=batch_size,
                          warp=warp,
                          hard_case_margin=hard_case_margin,
                          bias_init=bias_init, bias_range=bias_range, )
        self.K = K
        self.lambda_mean_distance = lambda_mean_distance
        self.lambda_variance = lambda_variance
        self.lambda_density = lambda_density
        self.lambda_cov = lambda_cov
        self.update_mu = update_mu
        self.update_density = update_density
        self.normalization = normalization
        self.variance_mu = variance_mu
        self.max_norm = max_norm

        # unused
        self.hard_case_chance = hard_case_chances

        if mixture_density is None:
            self.mixture_density = theano.shared(
                value=numpy.zeros((K, n_users)).astype(theano.config.floatX) + (1.0 / K),
                name='U_mixture_portion',
                borrow=True
            )
        else:
            self.mixture_density = theano.shared(
                value=mixture_density,
                name='U_mixture_portion',
                borrow=True
            )

        self.mixture_variance = theano.shared(
            value=numpy.zeros((K, n_users)).astype(theano.config.floatX) + self.variance_mu,
            name='variance',
            borrow=True
        )
        # self.item_variance = theano.shared(
        #     value=numpy.zeros((n_items, 1)).astype(theano.config.floatX) + 1.0,
        #     name='variance',
        #     borrow=True
        # )


        if normalization:
            self.V_norm = self.normalize(self.V)
            self.U_norm = self.normalize(self.U)
        else:
            self.V_norm = self.V
            self.U_norm = self.U

        self.U_norm_wide = self.U_norm.reshape((self.K, self.n_users, self.n_factors))
        self.mixture_variance_long = self.mixture_variance.reshape((self.K * self.n_users, 1))
        self.log_mixture_density_wide = T.log(self.mixture_density / self.mixture_density.sum(0)) #/
        self.log_mixture_density_long = self.log_mixture_density_wide.reshape((self.K * self.n_users, 1))

        if K == 1:
            self.pos_i_cluster = self.i
            self.neg_i_cluster = self.i
        else:
            self.pos_i_cluster = self.assign_cluster(self.i, self.j_pos)
            self.neg_i_cluster = self.assign_cluster(self.i, self.j_neg)
            one = numpy.asarray([1.0], dtype="float32").reshape((1, 1))
            self.user_sample_counts = T.inc_subtensor(
                T.zeros((self.n_users * self.K, 1), theano.config.floatX)[self.pos_i_cluster],
                one)
            self.user_sample_counts = T.inc_subtensor(self.user_sample_counts[self.neg_i_cluster], one)

        self.init_f = None

    def censor_updates(self):
        return super(KBPRModel, self).censor_updates() + [[self.V, self.max_norm], [self.U, self.max_norm]]

    def updates(self):

        updates = super(KBPRModel, self).updates()

        def fix(x):
            return T.maximum(1E-6, x)

        if self.update_density and self.K > 1:
            updates += [
                [self.mixture_density, self.user_sample_counts.reshape((self.K, self.n_users)), fix]]
        if self.update_mu:
            updates += [[self.mixture_variance, self.user_sample_counts.reshape((self.K, self.n_users))]]
        #updates += [[self.item_variance, self.item_variance]]
        return updates

    def cov_penalty(self, X):
        X = X - (X.sum(axis=0) / T.cast(X.shape[0], theano.config.floatX))
        return T.fill_diagonal(T.dot(X.T, X) / T.cast(X.shape[0], theano.config.floatX),
                               T.cast(0, theano.config.floatX))

    def cov_penalty_vectors(self):
        return [self.V, self.U]
    def l1_reg(self):
        reg = super(KBPRModel, self).l1_reg()
        if self.K > 1:
            reg += [[self.mixture_density / self.mixture_density.sum(0), self.lambda_density]]

        #reg += [[0 - self.scores_ij(self.i, self.j_pos), 0.001]]
        return reg
    def l2_reg(self):
        reg = super(KBPRModel, self).l2_reg()
        reg += [[(self.mixture_variance - self.variance_mu), self.lambda_variance]]
        #reg += [[(self.item_variance - self.variance_mu), self.lambda_variance]]

        if self.lambda_mean_distance != 0.0:
            center = T.concatenate([self.U_norm_wide.sum(0) / T.cast(self.K, "float32")] * self.K)
            reg += [[self.U - center, self.lambda_mean_distance]]

        reg += [[self.cov_penalty(T.concatenate(self.cov_penalty_vectors())),
                 self.lambda_cov]]
        return reg

    def assign_cluster(self, i, j):
        variance = self.mixture_variance[:, i] #* self.item_variance[j,0]
        distance = ((self.U_norm_wide[:, i, :] - self.V_norm[j]) ** 2).sum(axis=2)
        normal = -(distance / (2 * (variance ** 2))) - (
            T.log(variance ** 2) / 2)
        return (normal + self.log_mixture_density_wide[:, i]).argmax(
            axis=0) * self.n_users + i

    def factor_score_cluster_ij(self, i, j):
        distance = ((self.U_norm[i, :] - self.V_norm[j]) ** 2).sum(axis=1)
        distance_p = -(distance / (2 * (self.mixture_variance_long[i, 0] ** 2))) - (
            T.log(self.mixture_variance_long[i, 0] ** 2) / 2) + self.log_mixture_density_long[i, 0]
        return distance_p

    def factor_delta(self):
        pos_scores = self.factor_score_cluster_ij(self.pos_i_cluster, self.j_pos)
        neg_scores = self.factor_score_cluster_ij(self.neg_i_cluster, self.j_neg)
        return pos_scores - neg_scores

    def bias_delta(self):
        if self.use_bias:
            return T.log(self.b[self.j_pos, 0]) - T.log(self.b[self.j_neg, 0])
        return 0

    def scores_ij(self, i, j):
        i_cluster = self.assign_cluster(i, j)
        scores = self.factor_score_cluster_ij(i_cluster, j)
        if self.use_bias:
            scores += T.log(self.b[j, 0])
        return scores

    def bias_score(self):
        if self.use_bias:
            return T.log(self.b).reshape((self.n_items,))
        return 0

    def factor_score(self):
        variance = self.mixture_variance.reshape((1, self.K, self.n_users))
        if "item_variance" in self.__dict__:
            item_variance = self.item_variance.reshape((self.n_items, 1, 1))
        else:
            item_variance = 1.0 #numpy.zeros((self.n_items, 1, 1), dtype="float32") + 1
        portion = self.log_mixture_density_wide.reshape((1, self.K, self.n_users))
        v_norm_wide = self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))  # (items, 1, ,1, factors)
        distance = ((self.U_norm_wide[:, self.i, :] - v_norm_wide) ** 2).sum(axis=3)  # (items, K, users)
        normal = -(distance / (2 * ((variance[:, :, self.i] * item_variance) ** 2))) - (T.log((item_variance * variance[:, :, self.i]) ** 2) / 2)
        return (normal + portion[:, :, self.i]).max(axis=1).T  # transpose (items, users) to (users, items)

    def kmean(self, train_dict, K, user_batch=5000):
        from sklearn.utils.extmath import row_norms
        import numpy as np
        import scipy.sparse as sp
        from sklearn.metrics.pairwise import euclidean_distances
        def _k_init(X, n_local_trials=None):

            # from https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/cluster/k_means_.py#L43

            n_clusters = K
            random_state = numpy.random.RandomState(1)
            x_squared_norms = row_norms(X, squared=True)

            n_samples, n_features = X.shape

            centers = np.empty((n_clusters, n_features))

            assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

            # Set the number of local seeding trials if none is given
            if n_local_trials is None:
                # This is what Arthur/Vassilvitskii tried, but did not report
                # specific results for other than mentioning in the conclusion
                # that it helped.
                n_local_trials = 2 + int(np.log(n_clusters))

            # Pick first center randomly
            center_id = random_state.randint(n_samples)
            if sp.issparse(X):
                centers[0] = X[center_id].toarray()
            else:
                centers[0] = X[center_id]

            # Initialize list of closest distances and calculate current potential
            closest_dist_sq = euclidean_distances(
                centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
                squared=True)
            current_pot = closest_dist_sq.sum()

            # Pick the remaining n_clusters-1 points
            for c in range(1, n_clusters):
                # Choose center candidates by sampling with probability proportional
                # to the squared distance to the closest existing center
                rand_vals = random_state.random_sample(n_local_trials) * current_pot
                candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

                # Compute distances to center candidates
                distance_to_candidates = euclidean_distances(
                    X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

                # Decide which candidate is the best
                best_candidate = None
                best_pot = None
                best_dist_sq = None
                for trial in range(n_local_trials):
                    # Compute potential when including center candidate
                    new_dist_sq = np.minimum(closest_dist_sq,
                                             distance_to_candidates[trial])
                    new_pot = new_dist_sq.sum()

                    # Store result if it is the best local trial so far
                    if (best_candidate is None) or (new_pot < best_pot):
                        best_candidate = candidate_ids[trial]
                        best_pot = new_pot
                        best_dist_sq = new_dist_sq

                # Permanently add best center candidate found in local tries
                if sp.issparse(X):
                    centers[c] = X[best_candidate].toarray()
                else:
                    centers[c] = X[best_candidate]
                current_pot = best_pot
                closest_dist_sq = best_dist_sq

            return centers

        # variables
        prev_cost = theano.shared(numpy.cast['float32'](numpy.Infinity))
        all_clusters = []
        start = 0
        V = self.V.get_value()
        while True:
            end = min(start + user_batch, self.n_users)
            # positive samples
            tuples = []
            n_users = end - start
            init_clusters = numpy.zeros((n_users, K, self.n_factors))
            for u, items in train_dict.items():
                if start <= u < end:
                    # set initial centers with KMeans++
                    init_clusters[u - start, :, :] = _k_init(V[list(items)])
                    # add (user,item) pairs
                    for i in items:
                        tuples.append((u-start, i, 0))

            # theano
            clusters = theano.shared(numpy.transpose(init_clusters, [1, 0, 2]).astype("float32").reshape((n_users * K, self.n_factors)))
            clusters_norm_wide = clusters.reshape((K, n_users, self.n_factors))
            distance = ((clusters_norm_wide[:, self.i, :] - self.V_norm[self.j_pos]) ** 2).sum(axis=2)
            assign = distance.argmin(axis=0) * n_users + self.i

            cost = (((clusters[assign] - self.V_norm[self.j_pos]) ** 2).sum(axis=1) ).sum()
            new_clusters = T.inc_subtensor(T.zeros(clusters.shape)[assign], self.V_norm[self.j_pos])
            one = numpy.asarray([1.0], dtype="float32").reshape((1, 1))
            density = T.inc_subtensor(T.zeros((clusters.shape[0],1))[assign], one)
            new_clusters /= density + 1E-9

            f = theano.function([self.triplet], [cost, (prev_cost - cost) / prev_cost, density], updates=[
                [prev_cost, cost],
                [clusters, new_clusters]])
            i = 0
            while True:
                i += 1
                cur_cost, diff, assignments = f(tuples)
                print ("Iter %d, Cost %g, Converge %g" % (i, cur_cost, diff))
                if abs(diff) < 1e-6 and i>30:
                    break
            all_clusters.append(numpy.transpose(clusters.get_value().reshape((K, n_users, self.n_factors)),axes=[1,0,2]))
            start = end
            if start == self.n_users:
                break
        return numpy.transpose(numpy.concatenate(all_clusters), axes=[1,0,2]).reshape((K*self.n_users, self.n_factors))

    def wrong_hard_case_rate(self, valid_dict):
        fail = 0
        total = 0
        for u, cases in enumerate(self.hard_cases):
            if u in valid_dict:
                total += len([neg for _, neg, _ in cases.keys()])
                fail += len([neg for _, neg, _ in cases.keys() if neg in valid_dict[u]])
        return fail / float(total)

    def initialize(self, train_dict, epoch=1, adagrad=True, profile=False, exclude_items=None):
        if self.init_f is None:
            self.check_signature(train_dict)
            self.sample_generator = self.sample(train_dict, exclude_items)
            cost, per_sample_losses = self.cost()
            # do not update
            self.init_f = theano.function(
                inputs=[self.triplet, self.sample_weights, self.adagrad],
                outputs=[cost, per_sample_losses],
                updates=filter(lambda us: us[0] not in [self.V, self.b, ],
                               self.gen_updates(cost, per_sample_losses)),
                profile=profile
            )
        if self.warp is not False and self.warp_f is None:
            self.warp_f = self.warp_func()
        # perform latent vector updates #
        epoch_index = 0
        losses = []
        sample_time = 0
        training_time = 0
        sample_start = time.time()
        if adagrad:
            adagrad_val = 1
        else:
            adagrad_val = 0

        for triplet in self.sample_generator:
            sample_time += time.time() - sample_start
            # update latent vectors
            training_start = time.time()
            if self.warp is not False:
                triplet, weights = self.warp_f(triplet)
            else:
                weights = 1.0
            loss, per_sample_losses = self.init_f(triplet, weights, adagrad_val)
            print loss
            losses.append(loss)
            training_time += time.time() - training_start
            epoch_index += 1
            if epoch_index == epoch:
                sys.stderr.write(
                    "Train Time: %g Sample Time: %g Loss: %g\n" % (training_time, sample_time, numpy.mean(losses)))
                sys.stderr.flush()
                return numpy.mean(losses)
            sample_start = time.time()

    def __getstate__(self):
        ret = super(KBPRModel, self).__getstate__()
        # remove V_feature from the serialization
        ret["init_f"] = None
        return ret

    def params(self):
        return super(KBPRModel, self).params() + [
            ["margin", self.margin],
            ["K", self.K],
            ["l_m_d", self.lambda_mean_distance],
            ["l_var", self.lambda_variance],
            ["l_den", self.lambda_density],
            ["l_cov", self.lambda_cov],
            ["u_var", self.update_mu],
            ["u_den", self.update_density],
            ["chances", self.hard_case_chance],
            ["mu", self.variance_mu],
            ["norm", self.normalization],
            ["warp", self.warp],
            ["m_norm", self.max_norm]
        ]



class ContentBased(Model):
    def __init__(self, n_users, n_items, V_features, tfidf=False):

        super(ContentBased, self).__init__(n_users, n_items)
        self.tfidf = tfidf
        self.V_features = theano.shared(
            value=V_features.astype(theano.config.floatX),
            name='V_features',
            borrow=True
        )

        # User embedding (optional)
        self.U_features = None

        self.score_f = None
    def train(self, train_dict):
        V_features = self.V_features.get_value()
        U_features = numpy.zeros((self.n_users, V_features.shape[1])).astype(theano.config.floatX)
        for i in range(self.n_users):
            if i in train_dict:
                U_features[i] = V_features[list(train_dict[i])].sum(0)
        if self.tfidf:
            from sklearn.feature_extraction.text import TfidfTransformer
            transformer = TfidfTransformer()
            U_features = transformer.fit_transform(U_features).toarray()
        self.U_features = theano.shared(
            value=U_features.astype(theano.config.floatX),
            name='U_features',
            borrow=True
        )
        i = T.lvector()
        self.score_f = theano.function([i], T.dot(self.U_features[i], self.V_features.T) / T.sqrt(T.sqr(self.V_features).sum(1)).reshape((1, self.n_items)))
    def scores_for_users(self, users):
        return self.score_f(users)



class VisualKBPRAbstract(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U_features=None, items_with_features=None, U=None,
                 V=None, V_mlp=None,
                 U_mlp=None,
                 b=None, K=1, margin=1,
                 hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=True,
                 warp=True,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_mean_distance=0.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 lambda_cov=10,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.0000,
                 n_layers=2,
                 variance_mu=1.0,
                 update_mu=True,
                 update_density=True,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 per_user_sample=10,
                 normalization=False,
                 batch_size=-1,
                 max_norm=1,
                 embedding_rescale=0.04,
                 user_embedding_rescale=0.01,
                 width=128,
                 dropout_rate=0.5):

        super(VisualKBPRAbstract, self).__init__(n_factors, n_users, n_items,
                                                 update_mu=update_mu,
                                                 update_density=update_density,
                                                 normalization=normalization,
                                                 U=U, V=V, b=b, K=K, margin=margin, hard_case_margin=hard_case_margin,
                                                 mixture_density=mixture_density,
                                                 use_bias=use_bias,
                                                 warp=warp,
                                                 lambda_u=lambda_u,
                                                 lambda_v=lambda_v,
                                                 lambda_density=lambda_density,
                                                 variance_mu=variance_mu,
                                                 lambda_bias=lambda_bias,
                                                 lambda_mean_distance=lambda_mean_distance,
                                                 lambda_variance=lambda_variance,
                                                 uneven_sample=uneven_sample,
                                                 hard_case_chances=hard_case_chances,
                                                 learning_rate=learning_rate,
                                                 per_user_sample=per_user_sample,
                                                 batch_size=batch_size,
                                                 max_norm=max_norm,
                                                 lambda_cov=lambda_cov,

                                                 )
        self.width = width
        self.dropout_rate = dropout_rate
        import theano.sparse
        import scipy.sparse
        def to_tensor(m, name):
            if scipy.sparse.issparse(m):
                return theano.sparse.shared(
                    value=m,
                    name=name,
                    borrow=True
                    )
            else:
                return theano.shared(
                    value=m.astype(theano.config.floatX),
                    name=name,
                    borrow=True
                    )

        self.V_features = to_tensor(V_features, "V_features")
        # item embedding (optional)
        if items_with_features is None:
            self.items_with_features = numpy.arange(n_items)
        else:
            self.items_with_features = numpy.asarray(items_with_features, dtype="int64")

        if V_mlp is None:
            self.V_mlp = MLP(V_features.shape[1], self.n_factors, T.cast(self.n_items, "float32"),
                             lambda_weight_l1=lambda_weight_l1,
                             lambda_weight_l2=lambda_weight_l2, n_layers=n_layers, width=width)
        else:
            self.V_mlp = V_mlp.copy()
            self.V_mlp.usage_count = T.cast(self.n_items, "float32")
        self.embedding_rescale = embedding_rescale
        self.V_embedding = self.V_mlp.projection(self.V_features, self.embedding_rescale, self.max_norm,
                                                 dropout_rate=self.dropout_rate, training=True)

        # User embedding (optional)
        if U_features is not None:
            self.U_features = to_tensor(U_features, "U_features")

            if U_mlp is None:
                self.U_mlp = MLP(U_features.shape[1], n_factors, T.cast(self.n_users, "float32"),
                                 lambda_weight_l1=lambda_weight_l1,
                                 lambda_weight_l2=lambda_weight_l2, n_layers=n_layers, width=width)
            else:
                self.U_mlp = U_mlp.copy()
                self.U_mlp.usage_count = T.cast(self.n_users, "float32")
            self.user_embedding_rescale = user_embedding_rescale
            self.U_embedding = self.U_mlp.projection(self.U_features, self.user_embedding_rescale, self.max_norm,
                                                     dropout_rate=self.dropout_rate, training=True)
        else:
            self.U_features = None
            self.U_embedding = None
            self.U_mlp = None

    def l2_reg(self):
        reg = super(VisualKBPRAbstract, self).l2_reg()
        if self.U_mlp is not None:
            reg += self.U_mlp.l2_reg()
        reg += self.V_mlp.l2_reg()
        return reg

    def l1_reg(self):
        reg = super(VisualKBPRAbstract, self).l1_reg()
        if self.U_mlp is not None:
            reg += self.U_mlp.l1_reg()
        reg += self.V_mlp.l1_reg()
        return reg

    def __getstate__(self):
        ret = super(VisualKBPRAbstract, self).__getstate__()
        # remove V_feature from the serialization
        ret["V_features"] = None
        ret["V_embedding"] = None
        ret["U_features"] = None
        ret["U_embedding"] = None
        return ret



class VisualFactorKBPR(VisualKBPRAbstract):
    def __init__(self, n_factors, n_users, n_items, V_features, U_features=None, items_with_features=None, U=None,
                 V=None, V_mlp=None,
                 U_mlp=None, b=None, K=1, margin=1,
                 hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=True,
                 warp=True,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_mean_distance=0.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 lambda_cov=10,
                 n_layers=2,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.00001,
                 variance_mu=1.0,
                 update_mu=True,
                 update_density=True,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 per_user_sample=10,
                 normalization=False,
                 batch_size=-1,
                 max_norm=1,
                 lambda_v_off=0.1,
                 embedding_rescale=0.04,
                 user_embedding_rescale=0.01,
                 width=128,
                 dropout_rate=0.5):

        super(VisualFactorKBPR, self).__init__(n_factors, n_users, n_items, V_features, U_features,
                                               items_with_features=items_with_features,
                                               update_mu=update_mu,
                                               update_density=update_density,
                                               normalization=normalization,
                                               U=U, V=V, b=b, K=K, margin=margin,
                                               hard_case_margin=hard_case_margin,
                                               mixture_density=mixture_density,
                                               use_bias=use_bias,
                                               warp=warp,
                                               lambda_u=lambda_u,
                                               lambda_v=lambda_v,
                                               lambda_density=lambda_density,
                                               lambda_weight_l2=lambda_weight_l2,
                                               lambda_weight_l1=lambda_weight_l1,
                                               n_layers=n_layers,
                                               variance_mu=variance_mu,
                                               lambda_bias=lambda_bias,
                                               lambda_mean_distance=lambda_mean_distance,
                                               lambda_variance=lambda_variance,
                                               uneven_sample=uneven_sample,
                                               hard_case_chances=hard_case_chances,
                                               learning_rate=learning_rate,
                                               per_user_sample=per_user_sample,
                                               batch_size=batch_size,
                                               max_norm=max_norm,
                                               lambda_cov=lambda_cov,
                                               V_mlp=V_mlp,
                                               embedding_rescale=embedding_rescale,
                                               user_embedding_rescale=user_embedding_rescale,
                                               width=width,
                                               dropout_rate=dropout_rate,
                                               U_mlp=U_mlp)

        self.lambda_v_off = lambda_v_off
        self.update_f = None
        # initialize V as feature embedding value if V is not given
        if V is None:
            theano.function([], [], updates=[[self.V, self.V_embedding(numpy.arange(n_items))]])()
        if U is None and self.U_embedding is not None:
            embedding = self.U_embedding(numpy.arange(n_users))
            theano.function([], [], updates=[[self.U, T.concatenate([embedding] * self.K)]])()


    def cov_penalty_vectors(self):
        return super(VisualFactorKBPR, self).cov_penalty_vectors() #+ [self.V_embedding]

    def params(self):
        return super(VisualFactorKBPR, self).params() + [
            ["l_v_off", self.lambda_v_off]
        ]

    def updates(self):
        updates = super(VisualFactorKBPR, self).updates()
        updates += self.V_mlp.updates()
        if self.U_embedding is not None:
            updates += self.U_mlp.updates()

        return updates

    def l2_reg(self):
        reg = super(VisualFactorKBPR, self).l2_reg()
        has_features = (self.V_features ** 2).sum(1).nonzero()
        reg += [[(self.V - self.V_embedding(numpy.arange(self.n_items)))[has_features], self.lambda_v_off]]
        #reg += [[(self.U[self.i] - self.V_embedding[self.j_pos]), 0.0001, 1]]
        if self.U_embedding is not None:
            if self.K == 1:
                reg += [
                    [(self.U - self.U_embedding(numpy.arange(self.n_users))),
                     self.lambda_v_off]]
            else:

                embedding = self.U_embedding(numpy.arange(self.n_users))
                reg += [[(self.U - T.concatenate([embedding] * self.K)), self.lambda_v_off]]

            #reg += [[(self.U_embedding[self.i] - self.V_embedding[self.j_pos]), 0.0001, 1]]

        return reg

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False, profile=False, exclude_items=None):
        ret = super(VisualFactorKBPR, self).train(train_dict, epoch=epoch, adagrad=adagrad,
                                                  hard_case=hard_case, profile=profile, exclude_items=exclude_items)
        if exclude_items is not None and len(exclude_items) > 0:
            if self.update_f is None:
                print "Use non-dropout version"
                self.V_embedding = self.V_mlp.projection(self.V_features, self.embedding_rescale, self.max_norm,
                                                         dropout_rate=self.dropout_rate,
                                                         training=False)
                exclude_items = numpy.asarray(list(exclude_items))
                updates = [[self.V, T.set_subtensor(self.V[exclude_items], self.V_embedding[exclude_items])]]

                self.update_f = theano.function([], [], updates=updates)
            self.update_f()
        return ret

    def __getstate__(self):
        ret = super(VisualFactorKBPR, self).__getstate__()
        ret["update_f"] = None
        return ret


class VisualOnlyKBPR(VisualKBPRAbstract):
    def __init__(self, n_factors, n_users, n_items, V_features, items_with_features, U=None, V_mlp=None, b=None, K=1,
                 margin=1,
                 hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=True,
                 warp=True,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_mean_distance=0.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 lambda_cov=10,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.00001,
                 n_layers=2,
                 variance_mu=1.0,
                 update_mu=True,
                 update_density=True,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 per_user_sample=10,
                 normalization=False,
                 batch_size=-1,
                 max_norm=1,
                 lambda_v_off=0.1,
                 embedding_rescale=0.04):
        super(VisualOnlyKBPR, self).__init__(n_factors, n_users, n_items, V_features,
                                             items_with_features=items_with_features,
                                             update_mu=update_mu,
                                             update_density=update_density,
                                             normalization=normalization,
                                             U=U, V=None, b=b, K=K, margin=margin,
                                             hard_case_margin=hard_case_margin,
                                             mixture_density=mixture_density,
                                             use_bias=use_bias,
                                             warp=warp,
                                             lambda_u=lambda_u,
                                             lambda_v=lambda_v,
                                             lambda_density=lambda_density,
                                             lambda_weight_l2=lambda_weight_l2,
                                             lambda_weight_l1=lambda_weight_l1,
                                             n_layers=n_layers,
                                             variance_mu=variance_mu,
                                             lambda_bias=lambda_bias,
                                             lambda_mean_distance=lambda_mean_distance,
                                             lambda_variance=lambda_variance,
                                             uneven_sample=uneven_sample,
                                             hard_case_chances=hard_case_chances,
                                             learning_rate=learning_rate,
                                             per_user_sample=per_user_sample,
                                             batch_size=batch_size,
                                             max_norm=max_norm,
                                             lambda_cov=lambda_cov,
                                             V_mlp=V_mlp,
                                             embedding_rescale=embedding_rescale)

        self.lambda_v_off = lambda_v_off
        # initialize V as feature embedding value if V is not given
        self.V = self.V_embedding

    def updates(self):
        updates = super(VisualOnlyKBPR, self).updates()
        updates = filter(lambda u: u[0] != self.V, updates)
        updates += self.V_mlp.updates()
        return updates

    def l2_reg(self):
        reg = super(VisualOnlyKBPR, self).l2_reg()

        return reg


class MaxMF(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=2, margin=1, hard_case_margin=-1,
                 mixture_density=None,
                 use_bias=False,
                 warp=True,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_bias=0.1,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 uneven_sample=False,
                 per_user_sample=10,
                 lambda_cov=0.0,
                 max_norm=numpy.Infinity,
                 batch_size=-1):
        super(MaxMF, self).__init__(n_factors, n_users, n_items,
                                    update_mu=False,
                                    update_density=False,
                                    normalization=False,
                                    bias_init=0.0,
                                    bias_range=(-numpy.Infinity, numpy.Infinity),
                                    U=U, V=V, b=b, K=K, margin=margin, hard_case_margin=hard_case_margin,
                                    mixture_density=mixture_density,
                                    use_bias=use_bias,
                                    warp=warp,
                                    lambda_u=lambda_u,
                                    lambda_v=lambda_v,
                                    lambda_bias=lambda_bias,
                                    lambda_mean_distance=0.0,
                                    lambda_variance=0.0,
                                    uneven_sample=uneven_sample,
                                    hard_case_chances=hard_case_chances,
                                    learning_rate=learning_rate,
                                    per_user_sample=per_user_sample,
                                    batch_size=batch_size,
                                    max_norm=max_norm,
                                    lambda_cov=lambda_cov
                                    )

    def assign_cluster(self, i, j):
        return (self.U_norm_wide[:, i, :] * self.V_norm[j]).sum(axis=2).argmax(axis=0) * self.n_users + i


    def factor_delta(self):
        pos_i = self.assign_cluster(self.i, self.j_pos)
        neg_i = self.assign_cluster(self.i, self.j_neg)
        return ((self.U_norm[pos_i, :] * self.V_norm[self.j_pos]) -
                (self.U_norm[neg_i, :] * self.V_norm[self.j_neg])).sum(axis=1)

    def bias_delta(self):
        if self.use_bias:
            return self.b[self.j_pos, 0] - self.b[self.j_neg, 0]
        return 0

    def scores_ij(self, i, j):
        i_cluster = self.assign_cluster(i, j)
        scores = (self.U_norm[i_cluster, :] * self.V_norm[j]).sum(axis=1)
        if self.use_bias:
            scores += self.b[j, 0]
        return scores

    def bias_score(self):
        if self.use_bias:
            return self.b.reshape((self.n_items,))
        return 0

    def factor_score(self):
        # (items, K, users)
        scores = (self.U_norm_wide[:, self.i, :] * self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))).sum(
            axis=3)
        return scores.max(axis=1).T



class KNormalBPRModel(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=2, margin=1, hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=False,
                 warp=True,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_bias=0.1,
                 lambda_mean_distance=10.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 lambda_cov=0.1,
                 variance_mu=1.0,
                 bias_init=0.5,
                 update_mu=True,
                 update_density=True,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 per_user_sample=10,
                 normalization=True,
                 bias_range=(1E-6, 1),
                 batch_size=1000,
                 max_norm=1):

        super(KNormalBPRModel, self).__init__(n_factors, n_users, n_items,
                                              update_mu=update_mu,
                                              update_density=update_density,
                                              normalization=normalization,
                                              bias_init=bias_init,
                                              bias_range=bias_range,
                                              U=U, V=V, b=b, K=K, margin=margin, hard_case_margin=hard_case_margin,
                                              mixture_density=mixture_density,
                                              use_bias=use_bias,
                                              warp=warp,
                                              lambda_u=lambda_u,
                                              lambda_v=lambda_v,
                                              lambda_bias=lambda_bias,
                                              lambda_mean_distance=lambda_mean_distance,
                                              lambda_variance=lambda_variance,
                                              uneven_sample=uneven_sample,
                                              hard_case_chances=hard_case_chances,
                                              learning_rate=learning_rate,
                                              per_user_sample=per_user_sample,
                                              batch_size=batch_size,
                                              lambda_density=lambda_density,
                                              lambda_cov=lambda_cov,
                                              variance_mu=variance_mu,
                                              max_norm=max_norm)


    def prob(self, i, j, k):
        i = k * self.n_users + i
        log_density = self.log_mixture_density_long[i, 0]
        mu = (self.mixture_variance_long[i, 0])
        norm = T.exp(-((self.U_norm[i, :] - self.V_norm[j, :]) ** 2).sum(axis=1) / (2 * (mu ** 2)))
        norm /= mu
        norm *= T.exp(log_density)
        return norm

    def factor_score(self):
        mu = self.mixture_variance.reshape((1, self.K, self.n_users))[:, :, self.i]
        log_density = self.log_mixture_density_wide.reshape((1, self.K, self.n_users))[:, :, self.i]
        V = self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))  # (items, 1, ,1, factors)
        U = self.U_norm_wide[:, self.i, :]  # (K, users, factors)
        distance = ((U - V) ** 2).sum(axis=3)  # (items, K, users)
        return (T.exp(-distance / (2 * (mu ** 2))) / mu * T.exp(log_density)).sum(axis=1).T  # (user, items)

    def factor_delta(self):
        pos_distance = self.prob(self.i, self.j_pos, 0)
        for i in range(1, self.K):
            pos_distance += self.prob(self.i, self.j_pos, i)
        neg_distance = self.prob(self.i, self.j_neg, 0)
        for i in range(1, self.K):
            neg_distance += self.prob(self.i, self.j_neg, i)
        return T.log(pos_distance) - T.log(neg_distance)


class LightFMModel(Model):
    def __init__(self, n_factors, n_users, n_items, V_features=None, lambda_u=0.0, lambda_v=0.0, learning_rate=0.01,
                 loss="warp",
                 use_bias=True, normalize_features=False, exclude_items=None):

        super(LightFMModel, self).__init__(n_users, n_items)
        self.model = LightFM(learning_rate=learning_rate, loss=loss, no_components=n_factors, item_alpha=lambda_v,
                             user_alpha=lambda_u)
        self.train_coo = None
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.normalize_features = normalize_features
        from scipy.sparse import coo_matrix, csr_matrix
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)

        if V_features is not None:
            if normalize_features:
                V_features = V_features / ((1E-10 + V_features.sum(1).reshape((V_features.shape[0], 1))) ** 0.5)

            # exclude cold items' features from the features matrix and add one column at the end to each cold items
            # this make cold items looks "the same" to the algorithms and so prevent them from affecting the algorithm
            self.V_features_excluded = csr_matrix(V_features)
            if len(exclude_items) > 0:
                self.V_features_excluded[list(exclude_items), :] = 0.0
            m = coo_matrix(self.V_features_excluded)
            data = list(m.data)
            row = list(m.row)
            col = list(m.col)
            base = V_features.shape[1]
            for i in range(n_items):
                if i in exclude_items:
                    # set the last column to annotate cold item
                    row.append(i)
                    col.append(n_items + base)
                    data.append(1)
                else:
                    row.append(i)
                    col.append(i + base)
                    data.append(1)
            self.V_features_excluded = coo_matrix((data, (row, col)), shape=(n_items, n_items + base + 1))

            # include cold items' features in the features matrix so that the algorithm
            # can score the cold start recommendations
            self.V_features_orig = csr_matrix(V_features)
            m = coo_matrix(self.V_features_orig)
            data = list(m.data)
            row = list(m.row)
            col = list(m.col)
            base = V_features.shape[1]
            for i in range(n_items):
                if i not in exclude_items:
                    row.append(i)
                    col.append(i + base)
                    data.append(1)
            self.V_features_orig = coo_matrix((data, (row, col)), shape=(n_items, n_items + base + 1))

        else:
            self.V_features_excluded = None
            self.V_features_orig = None

    def params(self):
        return super(LightFMModel, self).params() + [
            ["lr", self.model.learning_rate],
            ["factors", self.model.no_components],
            ["l_u", self.model.user_alpha],
            ["l_v", self.model.item_alpha],
            ["loss", self.model.loss],
            ["norm_fea", self.normalize_features]

        ]

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False):
        if self.train_coo is None:
            self.check_signature(train_dict)
            self.train_coo = dict_to_coo(train_dict, self.n_users, self.n_items)
        import multiprocessing
        return self.model.fit_partial(self.train_coo, epochs=epoch, verbose=False,
                                      num_threads=multiprocessing.cpu_count(), item_features=self.V_features_excluded)

    def validate(self, train_dict, valid_dict, per_user_sample=100):
        return self.auc(valid_dict, train_dict)

    def scores_for_users(self, users):
        import multiprocessing
        for u in users:
            yield self.model.predict(u, numpy.arange(self.n_items),
                                     num_threads=multiprocessing.cpu_count(),
                                     item_features=self.V_features_orig)
