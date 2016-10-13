from sampler import create_sampler
import pyprind
import scipy
import numpy
from lightfm import LightFM
from utils import dict_to_coo
import theano
import time, sys
import theano.tensor as T
from theano.ifelse import ifelse
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


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
            ret = features  # features[j]
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
            embedding *= (desired_norms / (row_norms)).reshape((embedding.shape[0], 1))
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

    @staticmethod
    def normalize(variable):
        return variable / ((T.sum(variable ** 2, axis=1) ** 0.5).reshape((variable.shape[0], 1)))

    def check_signature(self, train_dict):
        signature = sum(map(lambda u: sum(u[1]) % (u[0] + 1), train_dict.items()))
        if self.signature is None or self.signature == signature:
            self.signature = signature
            return True
        raise Exception("Inconsistent train dict signature")

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

    def before_eval(self):
        return

    def auc(self, likes_dict, exclude_dict):
        self.before_eval()
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

        def arglargest_n(a, n):
            rev_a = -a
            ret = numpy.argpartition(rev_a, n)[:n]
            b = numpy.take(rev_a, ret)
            return numpy.take(ret, numpy.argsort(b))

        if exclude_items is not None:
            more_n = n + len(exclude_items)
        else:
            more_n = n
        for inx, scores in enumerate(self.scores_for_users(users)):
            user = users[inx]
            if user in exclude_dict:
                exclude = exclude_dict[user]
                local_n = more_n + len(exclude_dict[user])
                tops = arglargest_n(scores, local_n)[0:local_n]
            else:
                exclude = None
                tops = arglargest_n(scores, more_n)[0:more_n]
            tops = itertools.ifilter(lambda item: (exclude is None or item not in exclude)
                                                  and (exclude_items is None or item not in exclude_items),
                                     tops)
            yield itertools.islice(tops, n)

    def recall(self, likes_dict, exclude_dict, ns=(100, 50, 10), n_users=None, exclude_items=None, users=None):
        self.before_eval()
        from collections import defaultdict
        recall = defaultdict(list)

        if users is None:
            if n_users is None:
                users = likes_dict.keys()
            else:
                numpy.random.seed(1)
                users = numpy.random.choice(likes_dict.keys(), replace=False, size=n_users)
        if exclude_items is not None:
            exclude_items = set(exclude_items)
            selected_users = []
            for user in users:
                if len(likes_dict[user] - exclude_items) != 0:
                    selected_users.append(user)
            users = selected_users
        bar = pyprind.ProgBar(len(users))
        for inx, top in enumerate(self.topN(users, exclude_dict, n=max(ns), exclude_items=exclude_items)):
            user = users[inx]
            likes = likes_dict[user]
            top = list(top)
            for n in ns:
                hits = itertools.ifilter(lambda item: item in likes, itertools.islice(top, n))
                hits = reduce(lambda c, _: c + 1, hits, 0)
                recall[n].append(hits / float(len(likes)))
            bar.update(item_id=str(numpy.mean(recall[max(ns)])))
        return [[numpy.mean(recall[n]), scipy.stats.sem(recall[n])] for n in ns]

    def precision(self, likes_dict, exclude_dict, n=100, n_users=None, exclude_items=None):
        self.before_eval()
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

    def __getstate__(self):
        import copy
        ret = copy.copy(self.__dict__)
        return ret

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


class Popularity(Model):
    def __init__(self, n_users, n_items):
        Model.__init__(self, n_users, n_items)
        self.scores = None

    def train(self, train_dict):
        from collections import defaultdict
        counts = defaultdict(int)
        for items in train_dict.values():
            for item in items:
                counts[item] += 1
        self.scores = numpy.asarray([counts[i] for i in xrange(self.n_items)], dtype="float32")

    def scores_for_users(self, users):
        for u in users:
            yield numpy.copy(self.scores)


class ProfileKNN(Model):
    def __init__(self, n_users, n_items, U_features, K, active_users=None):
        Model.__init__(self, n_users, n_items)
        self.matrix = None
        self.profiles = U_features
        if active_users is None:
            self.active_users = numpy.arange(n_users)
        else:
            self.active_users = numpy.asarray(active_users)
        self.active_profile = self.profiles[active_users]
        self.K = K

    def train(self, train_dict):
        import scipy.sparse
        self.matrix = scipy.sparse.csr_matrix(dict_to_coo(train_dict, self.n_users, self.n_items))[self.active_users]

    def scores_for_users(self, users):

        sim_per_users = cosine_similarity(self.profiles[users, :], self.active_profile)
        for sim in sim_per_users:
            knn = numpy.argpartition(-sim, self.K)[0:self.K]
            yield (sim[knn] * self.matrix[knn])

    def params(self):
        return super(ProfileKNN, self).params() + [["K", self.K]]


class UserKNN(Model):
    def __init__(self, n_users, n_items, K, active_users=None, metric="cosine"):
        Model.__init__(self, n_users, n_items)
        self.matrix = None
        self.metric = metric
        if active_users is None:
            self.active_users = numpy.arange(n_users)
        else:
            self.active_users = numpy.asarray(active_users)
        self.active_users_matrix = None
        self.K = K

    def train(self, train_dict):
        import scipy.sparse
        self.matrix = scipy.sparse.csr_matrix(dict_to_coo(train_dict, self.n_users, self.n_items))
        self.active_users_matrix = self.matrix[self.active_users, :]

    def scores_for_users(self, users):
        for users in numpy.array_split(numpy.asarray(users, dtype="int32"),max(1, len(users) // 100)):
            if self.metric == "cosine":
                sim_per_users = cosine_similarity(self.matrix[users, :], self.active_users_matrix)
            else:
                sim_per_users = 1 - pairwise_distances(self.matrix[users, :], self.active_users_matrix, metric=self.metric)
            for sim in sim_per_users:
                knn = numpy.argpartition(-sim, self.K)[0:self.K]
                yield (sim[knn] * self.active_users_matrix[knn])

    def params(self):
        return super(UserKNN, self).params() + [["K", self.K]]


class BPRModel(Model):
    def __init__(self, n_factors, n_users, n_items,
                 U=None, V=None, b=None,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_b=0.1,
                 use_bias=True,
                 use_factors=True,
                 learning_rate=0.05,
                 loss_f="sigmoid",
                 margin=1,
                 warp_count=1,
                 batch_size=100000,
                 bias_init=0.0,
                 bias_range=(-numpy.Infinity, numpy.Infinity),

                 max_norm=numpy.Infinity,
                 negative_sample_choice="max"):

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
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.bias_range = bias_range
        self.warp_count = warp_count
        self.loss_f = loss_f
        self.margin = margin
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

        self.unique_j = T.arange(n_items)[self.item_sample_counts[:, 0].nonzero()]

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
        self.max_norm = max_norm
        self.negative_sample_choice = negative_sample_choice

    def params(self):
        return super(BPRModel, self).params() + [
            ["lr", self.learning_rate],
            ["use_f", self.use_factors],
            ["factors", self.n_factors],
            ["l_u", self.lambda_u],
            ["l_v", self.lambda_v],
            ["l_b", self.lambda_b],
            ["bias", self.use_bias],
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
        if "max_norm" in self.__dict__ and self.max_norm != numpy.Infinity:
            return [[self.V, self.max_norm], [self.U, self.max_norm]]
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
                cost += ((term ** 2)).sum() * l
        for term, l in self.l1_reg():
            if float(l) != float(0.0):
                cost += (abs(term)).sum() * l
        return cost

    # return a function that contains 1) the active samples in the given triplet array, 2) the weight assign to each
    # active sample. The active samples are the samples that, among the triplets that contains the same (user, pos_item)
    # pair, the triplet (user, pos_item, neg_item) that maximize f(user, neg_item) - f(user, pos_item).
    # the number of violations are used to estimate the rank of the pos_item to the user then the weight of the
    # active sample is computed as log of the estimated rank or zero if no violation is found.
    def warp_func(self):

        positive_pair_index = (T.arange(self.triplet.shape[0] / self.warp_count) * self.warp_count)
        # positive pair scores
        pos_scores = T.repeat(self.scores_ij(self.i[positive_pair_index], self.j_pos[positive_pair_index]),
                              self.warp_count)
        # negative pair scores (the number of negative pairs are "warp_count X" more than the positive pairs)
        neg_scores = self.scores_ij(self.i, self.j_neg)
        # apply hinge loss
        losses = T.maximum(0, self.margin - (pos_scores - neg_scores))
        # covert losses array to [n_pos_pairs, warp_count]
        losses = losses.reshape((losses.shape[0] / self.warp_count, self.warp_count))
        # reduce the violation counts of each positive pairs
        violations = (losses > 0).sum(1)
        # count weights for each positive pair (based on WASABIE)
        weights = T.switch(violations > 0, T.cast(T.log(self.n_items * violations / self.warp_count), "float32"),
                           T.zeros((violations.shape[0],), dtype="float32"))
        if "negative_sample_choice" not in self.__dict__ or self.negative_sample_choice == "max":
            # active triplet are those have maximun loss among the same positive pairs
            active_sample_index = T.argmax(losses, axis=1) + positive_pair_index
            active_samples = self.triplet[active_sample_index]
        else:
            weights = T.repeat(
                T.switch(violations > 0, T.cast(weights / violations, "float32"),
                         T.zeros((violations.shape[0],), dtype="float32")),
                self.warp_count
            )

            active_samples = self.triplet

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
            gradient = T.grad(cost=cost, wrt=param, disconnected_inputs='ignore') / (
                T.cast(dividend, "float32") + 1E-10)
            if regularization_cost != 0:
                gradient += T.grad(cost=regularization_cost, wrt=param, disconnected_inputs='ignore')
            new_history = ifelse(self.adagrad > 0, (history) + (gradient ** float(2)), history)
            update_list += [[history, new_history]]
            adjusted_grad = ifelse(self.adagrad > 0, gradient / ((new_history ** float(0.5)) + float(1e-10)), gradient)
            new_param = param - ((adjusted_grad) * float(self.learning_rate))
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
        return create_sampler(train_dict, self.n_items, self.warp_count, self.batch_size)

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False, profile=False, exclude_items=None):
        if self.warp_count > 1:
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
            if self.warp_count > 1:
                triplet, weights = self.warp_f(triplet)
            else:
                weights = numpy.zeros(len(triplet), dtype="float32") + 1.0
            # update latent vectors
            loss, per_sample_losses = self.f(triplet, weights, adagrad_val)
            losses.append(loss)
            training_time += time.time() - training_start
            epoch_index += 1
            if epoch_index == epoch:
                sys.stderr.write(
                    "Train Time: %g Sample Time: %g\n" % (training_time, sample_time))
                sys.stderr.flush()
                return numpy.mean(losses)
            sample_start = time.time()

    def __getstate__(self):
        ret = super(BPRModel, self).__getstate__()
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


class KBPRModel(BPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=1, margin=1,
                 mixture_density=None,
                 use_bias=True,
                 warp_count=10,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=10.0,
                 lambda_mean_distance=0.0,
                 lambda_variance=100.0,
                 lambda_density=1.0,
                 lambda_cov=0.0,
                 variance_mu=1.0,
                 bias_init=0.5,
                 update_mu=True,
                 update_density=True,
                 learning_rate=0.1,
                 normalization=False,
                 bias_range=(1E-6, 10),
                 batch_size=200000,
                 max_norm=1,
                 negative_sample_choice="max"):
        if U is None:
            U = numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users * K, n_factors)).astype(
                theano.config.floatX) / 5

        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_bias,
                          use_bias=use_bias,
                          use_factors=True,
                          loss_f="hinge",
                          margin=margin,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          warp_count=warp_count,
                          bias_init=bias_init,
                          bias_range=bias_range,
                          negative_sample_choice=negative_sample_choice,
                          max_norm=max_norm)
        self.K = K
        self.lambda_mean_distance = lambda_mean_distance
        self.lambda_variance = lambda_variance
        self.lambda_density = lambda_density
        self.lambda_cov = lambda_cov
        self.update_mu = update_mu
        self.update_density = update_density
        self.normalization = normalization
        self.variance_mu = variance_mu

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
        self.log_mixture_density_wide = T.log(self.mixture_density / self.mixture_density.sum(0))  # /
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

    def updates(self):

        updates = super(KBPRModel, self).updates()

        def fix(x):
            return T.maximum(1E-6, x)

        if self.update_density and self.K > 1:
            updates += [
                [self.mixture_density, self.user_sample_counts.reshape((self.K, self.n_users)), fix]]
        if self.update_mu:
            updates += [[self.mixture_variance, self.user_sample_counts.reshape((self.K, self.n_users))]]
        # updates += [[self.item_variance, self.item_variance]]
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

        # reg += [[0 - self.scores_ij(self.i, self.j_pos), 0.001]]
        return reg

    def l2_reg(self):
        reg = super(KBPRModel, self).l2_reg()
        reg += [[(self.mixture_variance - self.variance_mu), self.lambda_variance]]
        # reg += [[(self.item_variance - self.variance_mu), self.lambda_variance]]

        if self.lambda_mean_distance != 0.0:
            center = T.concatenate([self.U_norm_wide.sum(0) / T.cast(self.K, "float32")] * self.K)
            reg += [[self.U - center, self.lambda_mean_distance]]

        reg += [[self.cov_penalty(T.concatenate(self.cov_penalty_vectors())),
                 self.lambda_cov]]
        return reg

    def assign_cluster(self, i, j):
        variance = self.mixture_variance[:, i]  # * self.item_variance[j,0]
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
        item_variance = 1.0  # numpy.zeros((self.n_items, 1, 1), dtype="float32") + 1
        portion = self.log_mixture_density_wide.reshape((1, self.K, self.n_users))
        v_norm_wide = self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))  # (items, 1, ,1, factors)
        distance = ((self.U_norm_wide[:, self.i, :] - v_norm_wide) ** 2).sum(axis=3)  # (items, K, users)
        normal = -(distance / (2 * ((variance[:, :, self.i] * item_variance) ** 2))) - (
            T.log((item_variance * variance[:, :, self.i]) ** 2) / 2)
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
                        tuples.append((u - start, i, 0))

            # theano
            clusters = theano.shared(
                numpy.transpose(init_clusters, [1, 0, 2]).astype("float32").reshape((n_users * K, self.n_factors)))
            clusters_norm_wide = clusters.reshape((K, n_users, self.n_factors))
            distance = ((clusters_norm_wide[:, self.i, :] - self.V_norm[self.j_pos]) ** 2).sum(axis=2)
            assign = distance.argmin(axis=0) * n_users + self.i

            cost = (((clusters[assign] - self.V_norm[self.j_pos]) ** 2).sum(axis=1)).sum()
            new_clusters = T.inc_subtensor(T.zeros(clusters.shape)[assign], self.V_norm[self.j_pos])
            one = numpy.asarray([1.0], dtype="float32").reshape((1, 1))
            density = T.inc_subtensor(T.zeros((clusters.shape[0], 1))[assign], one)
            new_clusters /= density + 1E-9

            f = theano.function([self.triplet], [cost, (prev_cost - cost) / prev_cost, density], updates=[
                [prev_cost, cost],
                [clusters, new_clusters]])
            i = 0
            while True:
                i += 1
                cur_cost, diff, assignments = f(tuples)
                print ("Iter %d, Cost %g, Converge %g" % (i, cur_cost, diff))
                if abs(diff) < 1e-6 and i > 30:
                    break
            all_clusters.append(
                numpy.transpose(clusters.get_value().reshape((K, n_users, self.n_factors)), axes=[1, 0, 2]))
            start = end
            if start == self.n_users:
                break
        return numpy.transpose(numpy.concatenate(all_clusters), axes=[1, 0, 2]).reshape(
            (K * self.n_users, self.n_factors))

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
        if self.warp_count is not False and self.warp_f is None:
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
            if self.warp_count is not False:
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
            ["mu", self.variance_mu],
            ["norm", self.normalization],
            ["warp_count", self.warp_count],
            ["m_norm", self.max_norm]
        ]


class VisualKBPRAbstract(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U_features=None, items_with_features=None, U=None,
                 V=None, V_mlp=None,
                 U_mlp=None,
                 b=None, K=1, margin=1,
                 mixture_density=None,
                 use_bias=True,
                 warp_count=10,
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
                 learning_rate=0.01,
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
                                                 U=U, V=V, b=b, K=K, margin=margin,
                                                 mixture_density=mixture_density,
                                                 use_bias=use_bias,
                                                 warp_count=warp_count,
                                                 lambda_u=lambda_u,
                                                 lambda_v=lambda_v,
                                                 lambda_density=lambda_density,
                                                 variance_mu=variance_mu,
                                                 lambda_bias=lambda_bias,
                                                 lambda_mean_distance=lambda_mean_distance,
                                                 lambda_variance=lambda_variance,
                                                 learning_rate=learning_rate,
                                                 batch_size=batch_size,
                                                 max_norm=max_norm,
                                                 lambda_cov=lambda_cov,

                                                 )
        self.width = width
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        import theano.sparse
        import scipy.sparse
        def to_tensor(m, name):
            if scipy.sparse.issparse(m):
                return theano.sparse.shared(
                    value=scipy.sparse.csr_matrix(m, dtype="float32"),
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

    def params(self):
        return super(VisualKBPRAbstract, self).params() + [
            ["width", self.width],
            ["dropout_rate", self.dropout_rate],
            ["layers", self.n_layers]
        ]

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
                 mixture_density=None,
                 use_bias=True,
                 warp_count=10,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=10,
                 lambda_mean_distance=0.0,
                 lambda_variance=100.0,
                 lambda_density=1.0,
                 lambda_cov=0,
                 n_layers=2,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.00001,
                 variance_mu=1.0,
                 update_mu=True,
                 update_density=True,
                 learning_rate=0.1,
                 normalization=False,
                 batch_size=200000,
                 max_norm=1,
                 lambda_v_off=0.1,
                 embedding_rescale=0.04,
                 user_embedding_rescale=0.01,
                 width=256,
                 dropout_rate=0.5):

        super(VisualFactorKBPR, self).__init__(n_factors, n_users, n_items, V_features, U_features,
                                               items_with_features=items_with_features,
                                               update_mu=update_mu,
                                               update_density=update_density,
                                               normalization=normalization,
                                               U=U, V=V, b=b, K=K, margin=margin,
                                               mixture_density=mixture_density,
                                               use_bias=use_bias,
                                               warp_count=warp_count,
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

                                               learning_rate=learning_rate,
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
        return super(VisualFactorKBPR, self).cov_penalty_vectors()  # + [self.V_embedding]

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
        # has_features = (self.V_features ** 2).sum(1).nonzero()
        # reg += [[(self.V - self.V_embedding(numpy.arange(self.n_items)))[has_features], self.lambda_v_off]]
        reg += [[(self.V - self.V_embedding(numpy.arange(self.n_items))), self.lambda_v_off]]
        # reg += [[(self.U[self.i] - self.V_embedding[self.j_pos]), 0.0001, 1]]
        if self.U_embedding is not None:
            if self.K == 1:
                reg += [
                    [(self.U - self.U_embedding(numpy.arange(self.n_users))),
                     self.lambda_v_off]]
            else:

                embedding = self.U_embedding(numpy.arange(self.n_users))
                reg += [[(self.U - T.concatenate([embedding] * self.K)), self.lambda_v_off]]

                # reg += [[(self.U_embedding[self.i] - self.V_embedding[self.j_pos]), 0.0001, 1]]

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
        self.n_factors = n_factors
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.loss = loss
        self.U = None
        self.V = None
        self.score_f = None
        self.with_features = V_features is not None
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
        self.V = self.U = self.score_f = None
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
        if "U" not in self.__dict__ or self.U is None:
            if self.V_features_orig is not None:
                V = self.V_features_orig.dot(self.model.item_embeddings)
                bias = self.V_features_orig.dot(self.model.item_biases)
            else:
                V = self.model.item_embeddings
                bias = self.model.item_biases
            self.V = theano.shared(numpy.concatenate(
                (V, bias.reshape((self.n_items, 1))), axis=1))
            self.U = theano.shared(numpy.concatenate(
                (self.model.user_embeddings, numpy.ones((self.n_users, 1), dtype="float32")), axis=1))
            u = theano.tensor.iscalar()
            self.score_f = theano.function([u], T.dot(self.U[u], self.V.T))
        for u in users:
            # group_truth = self.model.predict(u, numpy.arange(self.n_items),
            #                          num_threads=multiprocessing.cpu_count(),
            #                          item_features=self.V_features_orig)
            # error = numpy.nonzero(numpy.argsort(numpy.argsort(group_truth)) - numpy.argsort(numpy.argsort(scores)))
            yield self.score_f(u)

    def __getstate__(self):
        ret = super(LightFMModel, self).__getstate__()
        ret["U"] = None
        ret["V"] = None
        ret["score_f"] = None

        return ret


class BPRModelWithUserProfile(BPRModel):
    def __init__(self, n_factors, n_users, n_items, U_features, n_active_user, U=None, V=None, b=None, U_mlp=None,
                 margin=1,
                 use_bias=True,
                 warp_count=20,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.0,
                 bias_init=0.0,
                 lambda_weight_l1=0,
                 lambda_weight_l2=0,
                 n_layers=2,
                 width=256,
                 learning_rate=0.1,
                 bias_range=(-numpy.Infinity, numpy.Infinity),
                 user_embedding_rescale=1.0,
                 batch_size=200000,
                 negative_sample_choice="max",
                 dropout_rate=0.5,
                 lambda_u_off=1):
        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_bias,
                          use_bias=use_bias,
                          use_factors=True,
                          loss_f="hinge",
                          margin=margin,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          warp_count=warp_count,
                          bias_init=bias_init,
                          bias_range=bias_range,
                          negative_sample_choice=negative_sample_choice,
                          )

        # User embedding (optional)
        def to_tensor(m, name):
            if scipy.sparse.issparse(m):
                return theano.sparse.shared(
                    value=scipy.sparse.csr_matrix(m, dtype="float32"),
                    name=name,
                    borrow=True
                )
            else:
                return theano.shared(
                    value=m.astype(theano.config.floatX),
                    name=name,
                    borrow=True
                )

        self.n_active_user = n_active_user
        self.U_features = to_tensor(U_features, "U_features")
        self.n_layers = n_layers
        self.width = width
        if U_mlp is None:
            self.U_mlp = MLP(U_features.shape[1], n_factors, T.cast(self.n_users, "float32"),
                             lambda_weight_l1=lambda_weight_l1,
                             lambda_weight_l2=lambda_weight_l2, n_layers=n_layers, width=width)
        else:
            self.U_mlp = U_mlp.copy()
            self.U_mlp.usage_count = T.cast(self.n_users, "float32")
        self.user_embedding_rescale = user_embedding_rescale
        self.dropout_rate = dropout_rate
        self.U_embedding = self.U_mlp.projection(self.U_features, self.user_embedding_rescale, numpy.Infinity,
                                                 dropout_rate=self.dropout_rate, training=True)
        self.U_embedding_full = self.U_mlp.projection(self.U_features, self.user_embedding_rescale, numpy.Infinity,
                                                      dropout_rate=1, training=False)
        self.lambda_u_off = lambda_u_off
        self.update_f = None
        if U is None and self.U_embedding is not None:
            embedding = self.U_embedding(numpy.arange(n_users))
            theano.function([], [], updates=[[self.U, embedding]])()

    def params(self):
        return super(BPRModelWithUserProfile, self).params() + [
            ["width", self.width],
            ["dropout_rate", self.dropout_rate],
            ["layers", self.n_layers],
            ["l_u_off", self.lambda_u_off]
        ]

    def updates(self):
        updates = super(BPRModelWithUserProfile, self).updates()
        updates += self.U_mlp.updates()
        return updates

    def l2_reg(self):
        reg = super(BPRModelWithUserProfile, self).l2_reg()
        reg += [[(self.U - self.U_embedding(numpy.arange(self.n_users)))[0:self.n_active_user], self.lambda_u_off]]
        reg += self.U_mlp.l2_reg()
        return reg

    def l1_reg(self):
        reg = super(BPRModelWithUserProfile, self).l1_reg()
        reg += self.U_mlp.l1_reg()
        return reg

    def before_eval(self):
        super(BPRModelWithUserProfile, self).before_eval()
        if self.U_embedding is not None:
            if self.update_f is None:
                print "Use non-dropout version"
                updates = [[self.U, T.set_subtensor(self.U[self.n_active_user:],
                                                    self.U_embedding_full(numpy.arange(self.n_users))[
                                                    self.n_active_user:])]]
                self.update_f = theano.function([], [], updates=updates)
            self.update_f()
        return

    def __getstate__(self):
        ret = super(BPRModelWithUserProfile, self).__getstate__()
        # remove V_feature from the serialization
        ret["V_features"] = None
        ret["V_embedding"] = None
        ret["U_features"] = None
        ret["U_embedding"] = None
        ret["update_f"] = None
        ret["U_embedding_full"] = None

        return ret


class VisualOffsetBPR(BPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U_features=None, items_with_features=None, U=None,
                 V=None, V_mlp=None,
                 U_mlp=None,
                 b=None, margin=1,
                 use_bias=True,
                 warp_count=10,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.0000,
                 n_layers=2,
                 learning_rate=0.01,
                 batch_size=200000,
                 max_norm=numpy.Inf,
                 embedding_rescale=0.04,
                 user_embedding_rescale=0.01,
                 width=128,
                 dropout_rate=0.5,
                 loss_f="hinge",
                 lambda_v_off=1):

        super(VisualOffsetBPR, self).__init__(n_factors, n_users, n_items,
                                              U=U, V=V, b=b, margin=margin,
                                              use_bias=use_bias,
                                              warp_count=warp_count,
                                              lambda_u=lambda_u,
                                              lambda_v=lambda_v,
                                              learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              max_norm=max_norm,
                                              loss_f=loss_f,
                                              lambda_b=lambda_bias,

                                              )
        self.width = width
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.lambda_v_off = lambda_v_off
        import theano.sparse
        import scipy.sparse
        def to_tensor(m, name):
            if scipy.sparse.issparse(m):
                return theano.sparse.shared(
                    value=scipy.sparse.csr_matrix(m, dtype="float32"),
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
        reg = super(VisualOffsetBPR, self).l2_reg()
        if self.U_mlp is not None:
            reg += self.U_mlp.l2_reg()
        reg += self.V_mlp.l2_reg()
        reg += [[(self.V - self.V_embedding(numpy.arange(self.n_items))), self.lambda_v_off]]
        if self.U_embedding is not None:
            reg += [
                [(self.U - self.U_embedding(numpy.arange(self.n_users))),
                 self.lambda_v_off]]

        return reg

    def l1_reg(self):
        reg = super(VisualOffsetBPR, self).l1_reg()
        if self.U_mlp is not None:
            reg += self.U_mlp.l1_reg()
        reg += self.V_mlp.l1_reg()
        return reg

    def params(self):
        return super(VisualOffsetBPR, self).params() + [
            ["width", self.width],
            ["dropout_rate", self.dropout_rate],
            ["layers", self.n_layers]
        ]

    def updates(self):
        updates = super(VisualOffsetBPR, self).updates()
        updates += self.V_mlp.updates()
        if self.U_embedding is not None:
            updates += self.U_mlp.updates()

        return updates

    def __getstate__(self):
        ret = super(VisualOffsetBPR, self).__getstate__()
        # remove V_feature from the serialization
        ret["V_features"] = None
        ret["V_embedding"] = None
        ret["U_features"] = None
        ret["U_embedding"] = None
        return ret



class VisualBPR(BPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features,  items_with_features=None, U=None,
                 V=None, V_mlp=None,

                 b=None, margin=1,
                 use_bias=True,
                 warp_count=10,
                 lambda_u=0.0,
                 lambda_v=0.0,
                 lambda_bias=0.1,
                 lambda_weight_l2=0.00001,
                 lambda_weight_l1=0.0000,
                 n_layers=2,
                 learning_rate=0.01,
                 batch_size=200000,
                 max_norm=numpy.Inf,
                 embedding_rescale=0.04,
                 width=128,
                 dropout_rate=0.5,
                 loss_f="hinge"):

        super(VisualBPR, self).__init__(n_factors, n_users, n_items,
                                              U=U, V=V, b=b, margin=margin,
                                              use_bias=use_bias,
                                              warp_count=warp_count,
                                              lambda_u=lambda_u,
                                              lambda_v=lambda_v,
                                              learning_rate=learning_rate,
                                              batch_size=batch_size,
                                              max_norm=max_norm,
                                              loss_f=loss_f,
                                              lambda_b=lambda_bias,

                                              )
        self.width = width
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        import theano.sparse
        import scipy.sparse
        def to_tensor(m, name):
            if scipy.sparse.issparse(m):
                return theano.sparse.shared(
                    value=scipy.sparse.csr_matrix(m, dtype="float32"),
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

        self.U_Visual = theano.shared(
            value=numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users, n_factors)).astype(
                theano.config.floatX) / 5,
            name='U',
            borrow=True
        )

        self.embedding_rescale = embedding_rescale
        self.V_embedding = self.V_mlp.projection(self.V_features, self.embedding_rescale, self.max_norm,
                                                 dropout_rate=self.dropout_rate, training=True)(numpy.arange(self.n_items))

    def l2_reg(self):
        reg = super(VisualBPR, self).l2_reg()
        reg += self.V_mlp.l2_reg()
        if self.lambda_u != 0.0:
            reg.append((self.U_Visual, self.lambda_u))
        if self.lambda_v != 0.0:
            reg.append((self.V_embedding, self.lambda_v))

        return reg

    def l1_reg(self):
        reg = super(VisualBPR, self).l1_reg()
        reg += self.V_mlp.l1_reg()
        return reg

    def params(self):
        return super(VisualBPR, self).params() + [
            ["width", self.width],
            ["dropout_rate", self.dropout_rate],
            ["layers", self.n_layers]

        ]

    def updates(self):
        updates = super(VisualBPR, self).updates()
        updates += self.V_mlp.updates()
        updates += [[self.U_Visual, self.user_sample_counts]]
        return updates

    def factor_delta(self):
        # user vector
        delta = super(VisualBPR, self).factor_delta() + \
                (self.U_Visual[self.i] * (self.V_embedding[self.j_pos] - self.V_embedding[self.j_neg])).sum(axis=1)
        return delta

    def scores_ij(self, i, j):
        return super(VisualBPR, self).scores_ij(i, j) + (self.U_Visual[i] * (self.V_embedding[j])).sum(axis=1)

    def factor_score(self):
        return super(VisualBPR, self).factor_score() + T.dot(self.U_Visual[self.i], self.V_embedding.T)

    def __getstate__(self):
        ret = super(VisualBPR, self).__getstate__()
        # remove V_feature from the serialization
        if not isinstance(self.V_embedding, numpy.ndarray):
            ret["V_embedding"] = theano.function([], self.V_embedding)()
        ret["V_features"] = None
        return ret
