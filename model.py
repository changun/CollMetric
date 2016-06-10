import pyximport;

pyximport.install()
import fast_utils
import copy
import heapq
import numpy
import pyprind
import random
import scipy
import sys
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import time
from collections import defaultdict
from theano.ifelse import ifelse
from utils import *
from lightfm import LightFM
from multiprocessing import Pool
import gc
import signal

gen = {}
pool = Pool(processes=1)
def sampler(id):
    if id in gen:
        return gen[id].next()
    else:
        return None

def init_sampler(id, train_dict, exclude_dict, n_items, per_user_sample):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    gen[id] = fast_utils.sample(train_dict, exclude_dict, n_items, per_user_sample)

class Model(object):
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    def normalize(self, variable):
        return variable / ((T.sum(variable ** 2, axis=1) ** 0.5).reshape((variable.shape[0], 1)))

    def copy(self):
        return {
            "n_users": self.n_users,
            "n_items": self.n_items
        }

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

    def topN(self, users, exclude_dict, n=100):
        for inx, scores in enumerate(self.scores_for_users(users)):
            user = users[inx]
            if exclude_dict is not None and user in exclude_dict:
                scores[list(exclude_dict[user])] = -numpy.Infinity
            yield numpy.argpartition(-scores, n)[0:n]

    def recall(self, likes_dict, exclude_dict, n=100, n_users=None):
        recall = []
        try:
            if n_users is None:
                users = likes_dict.keys()
            else:
                numpy.random.seed(1)
                users = numpy.random.choice(likes_dict.keys(), replace=False, size=n_users)
            bar = pyprind.ProgBar(len(users))
            for inx, top in enumerate(self.topN(users, exclude_dict, n=n)):
                user = users[inx]
                likes = likes_dict[user]
                hits = [j for j in top if j in likes]
                recall.append(len(hits) / float(len(likes)))
                bar.update(item_id=str(numpy.mean(recall)))
        except Exception as e:
            sys.stderr.write(str(e))
        finally:
            return numpy.mean(recall), scipy.stats.sem(recall), recall

    def precision(self, likes_dict, exclude_dict, n=100, n_users=None):
        precision = []
        try:
            if n_users is None:
                users = likes_dict.keys()
            else:
                numpy.random.seed(1)
                users = numpy.random.choice(likes_dict.keys(), replace=False, size=n_users)
            bar = pyprind.ProgBar(len(users))
            for inx, top in enumerate(self.topN(users, exclude_dict, n=n)):
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
                 learning_rate=0.05, per_user_sample=10, bias_init=0.0, bias_range=(-numpy.Infinity, numpy.Infinity)):

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
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.bias_range = bias_range

        if use_factors:
            if U is None:
                self.U = theano.shared(
                    value=numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users, n_factors)).astype(
                        theano.config.floatX),
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
                        theano.config.floatX),
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

        # user id
        self.triplet = T.lmatrix('triplet')
        self.i = self.triplet[:, 0]

        # a vector of pos item ids
        self.j_pos = self.triplet[:, 1]
        # a vector of neg item ids
        self.j_neg = self.triplet[:, 2]

        self.unique_j = T.lvector()
        # for those samples that do not appear in the triplet, set the count to -1
        minus1 = T.zeros((self.n_items, 1), theano.config.floatX)-1
        # for those samples that do appear in the triplet, set the count to 0 first

        one = numpy.asarray([1.0], dtype="float32").reshape((1,1))

        self.item_sample_counts_with_minus = T.inc_subtensor(minus1[self.unique_j], one)
        # add counts
        self.item_sample_counts_with_minus = T.inc_subtensor(
            T.inc_subtensor(self.item_sample_counts_with_minus[self.j_pos, :], one)[self.j_neg, :], one)
        # add counts
        self.item_sample_counts = T.inc_subtensor(
            T.inc_subtensor(T.zeros((self.n_items, 1), theano.config.floatX)[self.j_pos, :], one)[self.j_neg, :], one)


        self.unique_i =  T.lvector()
        # for those samples that do not appear in the triplet, set the count to -1
        minus1 = T.zeros((self.n_users, 1), theano.config.floatX) - 1
        # for those samples that do appear in the triplet, set the count to 0 first
        self.user_sample_counts_with_minus = T.inc_subtensor(minus1[self.unique_i, :], one)
        # add counts
        self.user_sample_counts_with_minus = T.inc_subtensor(self.user_sample_counts_with_minus[self.i, :], one)
        # add counts
        self.user_sample_counts = T.inc_subtensor(T.zeros((self.n_users, 1), theano.config.floatX)[self.i, :], one)


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

        self.f = None
        self.global_f = None
        self.score_f = None
        self.validate_f = None
        self.cache_f = None

    def copy(self):
        z = super(BPRModel, self).copy()
        z.update(
            {"n_factors": self.n_factors,
             "lambda_v": self.lambda_v,
             "lambda_u": self.lambda_u,
             "use_factors": self.use_factors,
             "use_bias": self.use_bias,
             "V": self.V.get_value(),
             "U": self.U.get_value(),
             "lambda_b": self.lambda_b,
             "b": self.b.get_value(),
             "learning_rate": self.learning_rate})
        return z

    def params(self):
        return super(BPRModel, self).params() + [
            ["lr", self.learning_rate],
            ["use_f", self.use_factors],
            ["factors", self.n_factors],
            ["l_u", self.lambda_u],
            ["l_v", self.lambda_v],
            ["l_b", self.lambda_b],
            ["per_u", self.per_user_sample],
            ["bias", self.use_bias]
        ]

    def l2_reg(self):
        return [(self.U, self.lambda_u, self.user_sample_counts),
                (self.V, self.lambda_v, self.item_sample_counts),
                (self.b - self.bias_init, self.lambda_b, self.item_sample_counts)
                ]

    def l1_reg(self):
        return []

    def updates(self):
        updates = []
        if self.use_factors:
            updates += [[self.U, self.user_sample_counts_with_minus],
                        # self.item_sample_counts_with_minus set -1 for those items that DO NOT appear in the triplet
                        # the gradient for those items should be zeros, we do so to avoid 0/0 = NAN issue
                        [self.V, self.item_sample_counts_with_minus],
                        ]

        if self.use_bias:
            def limit(b): return T.minimum(T.maximum(b, self.bias_range[0]), self.bias_range[1])
            updates += [[self.b, self.item_sample_counts_with_minus, limit]]
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
        f = theano.function(
            inputs=[self.i],
            outputs=self.score_fn()
        )
        for users in numpy.array_split(numpy.asarray(users, dtype="int32"), len(users) // 4):
            for scores in f(users):
                yield scores

    def loss(self, delta):
        return -T.log(T.nnet.sigmoid(delta))

    def cost(self):
        # user vector
        delta = self.delta()
        losses = self.loss(delta)

        cost = losses.sum()
        for term, l, multiply in self.l2_reg():
            cost += ((term ** 2) * multiply).sum() * l
        for term, l in self.l1_reg():
            cost += (abs(term)).sum() * l
        return cost, losses,

    def sample(self, train_dict, exclude_dict=None, per_user_sample=None):
        if per_user_sample is None:
            per_user_sample = self.per_user_sample

        id = sum([ u * sum(items) for u, items in train_dict.items()])
        if exclude_dict is not None:
            id += sum([u * sum(items) for u, items in exclude_dict.items()])
        id *= per_user_sample
        samples = pool.apply_async(sampler, (id, )).get()
        if samples is None:
            pool.apply_async(init_sampler, (id, train_dict, exclude_dict, self.n_items, per_user_sample)).get()
        res = pool.apply_async(sampler, (id,))
        try:
            while True:
                samples = res.get()
                res = pool.apply_async(sampler, (id, ))
                yield samples

        finally:
            print "clear sampler!"






    def gen_updates(self, cost, gen_updates):
        update_list = []
        for update in self.updates():
            param = update[0]
            dividend = update[1]

            history = theano.shared(
                value=numpy.zeros(param.get_value().shape).astype(theano.config.floatX),
                borrow=True
            )
            if isinstance(dividend, int) or isinstance(dividend, long) or isinstance(dividend, float):
                gradient = T.grad(cost=cost, wrt=param) / float(dividend)
            else:
                gradient = T.grad(cost=cost, wrt=param) / dividend
            new_history = history + (gradient ** float(2))
            update_list += [[history, new_history]]
            adjusted_grad = ifelse(self.adagrad > 0, gradient / ((new_history ** float(0.5)) + float(1e-10)), gradient)
            new_param = param - (adjusted_grad * float(self.learning_rate))
            if len(update) == 3:
                new_param = update[2](new_param)
            for censored_param, limit in self.censor_updates():
                if param == censored_param:
                    col_norms = T.sqrt(T.sum(T.sqr(new_param), axis=1))
                    desired_norms = T.clip(col_norms, 0, limit)
                    new_param *= (desired_norms / (1e-7 + col_norms)).reshape((new_param.shape[0], 1))
            update_list += [[param, new_param]]
        return update_list

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False):
        if hard_case:
            raise Exception("Hard case sampling is not supported")
        if self.train_dict != train_dict:
            self.train_dict = train_dict
            self.sample_generator = self.sample(train_dict)
        cost, per_sample_loss = self.cost()
        if self.f is None:
            self.f = theano.function(
                inputs=[self.triplet, self.adagrad, self.unique_i, self.unique_j],
                outputs=[cost, per_sample_loss],
                updates=self.gen_updates(cost, per_sample_loss)
            )
        # perform latent vector updates #
        epoch_index = 0
        loss = []
        sample_time = 0
        training_time = 0
        sample_start = time.time()
        if adagrad:
            adagrad_val = 1
        else:
            adagrad_val = 0
        for triplet, unique_i, unique_j in self.sample_generator:
            sample_time += time.time() - sample_start
            # update latent vectors
            training_start = time.time()
            loss.append(self.f(triplet, adagrad_val, unique_i, unique_j)[0])
            training_time += time.time() - training_start
            epoch_index += 1
            if epoch_index == epoch:
                sys.stderr.write(
                    "Train Time: %g Sample Time: %g\n" % (training_time, sample_time))
                sys.stderr.flush()
                return numpy.mean(loss)
            sample_start = time.time()

    def validate(self, train_dict, valid_dict, per_user_sample=100):
        if self.validate_f is None:
            # compute AUC
            delta = self.delta()
            self.validate_f = theano.function(
                inputs=[self.triplet],
                outputs=T.switch(T.gt(delta, 0), 1.0, 0.0)
            )
            self.valid_sample_generator = self.sample(valid_dict, train_dict, per_user_sample=per_user_sample)
            self.train_valid_sample_generator = self.sample(train_dict, None, per_user_sample=per_user_sample)

        results = None
        if train_dict == None:
            gen = self.train_valid_sample_generator
        else:
            gen = self.valid_sample_generator
        # sample triplet with the sample generator
        for triplet, unique_i, unique_j in gen:
            # update latent vectors
            results = self.validate_f(triplet)
            break
        results = results.reshape((len(results) / per_user_sample, per_user_sample))
        aucs = numpy.mean(results, axis=1)
        return numpy.mean(aucs), scipy.stats.sem(aucs)


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
            self.weights = [theano.shared(numpy.asarray(
                w,
                dtype=theano.config.floatX
            ), borrow=True)
                            for w in weights
                            ]
        else:
            if self.nonlinear:
                self.weights = [
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
                    self.weights += [
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

                self.weights += [theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (128 + n_embedding_dim)),
                        high=numpy.sqrt(6. / (128 + n_embedding_dim)),
                        size=(128, n_embedding_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True, name='weights_end')]
            # linear embedding
            else:
                self.weights = [theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                        high=numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                        size=(raw_visual_feature_factors, n_embedding_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True, name='weights')]

        self.V_embedding = self.embedding(self.V_features, self.weights)

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

            self.V_embedding = self.embedding(to_T(item_features), self.weights)
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
             "weights": [w.get_value() for w in self.weights],
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
        reg = super(VisualBPRAbstractModel, self).l2_reg() + [[w, self.lambda_weight_l2] for w in self.weights]
        if self.use_visual_offset:
            reg += [[self.V_offset, self.lambda_v_offset]]
        return reg

    def l1_reg(self):
        return super(VisualBPRAbstractModel, self).l1_reg() + [[w, self.lambda_weight_l1] for w in self.weights]

    def updates(self):
        updates = super(VisualBPRAbstractModel, self).updates()

        updates += [
            [w, self.n_users * self.per_user_sample]
            for w in self.weights + self.u_weights]

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
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=2, margin=1, hard_case_margin=-1,
                 mixture_density=None,
                 uneven_sample=False,
                 use_bias=False,
                 use_warp=True,
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
        if U is None:
            U = numpy.random.normal(0, 1 / (n_factors ** 0.5), (n_users * K, n_factors)).astype(theano.config.floatX)

        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_bias,
                          use_bias,
                          True,
                          learning_rate,
                          per_user_sample,
                          bias_init=bias_init, bias_range=bias_range,)
        self.K = K
        self.margin = margin
        self.lambda_mean_distance = lambda_mean_distance
        self.lambda_variance = lambda_variance
        self.lambda_density = lambda_density
        self.lambda_cov = lambda_cov
        self.update_mu = update_mu
        self.hard_case_chance = hard_case_chances
        self.use_warp = use_warp
        self.uneven_sample = uneven_sample
        self.update_density = update_density
        self.normalization = normalization
        self.hard_cases = [dict() for _ in range(n_users)]
        self.hard_case_margin = hard_case_margin
        self.pos_sample_count = None
        self.variance_mu = variance_mu
        self.trials = None
        self.batch_size = batch_size
        self.max_norm = max_norm
        if mixture_density is None:
            self.mixture_density = theano.shared(
                value=numpy.zeros((K - 1, n_users)).astype(theano.config.floatX) + (1.0 / K),
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

        if normalization:
            self.V_norm = self.normalize(self.V)
            self.U_norm = self.normalize(self.U)
        else:
            self.V_norm = self.V
            self.U_norm = self.U

        self.U_norm_wide = self.U_norm.reshape((self.K, self.n_users, self.n_factors))
        self.mixture_variance_long = self.mixture_variance.reshape((self.K * self.n_users, 1))
        first_mixture_portion = T.maximum(0, (1 - T.maximum(0, self.mixture_density).sum(axis=0))).reshape(
            (1, n_users))
        self.log_mixture_density_wide = T.log(
            T.concatenate((first_mixture_portion, T.maximum(0, self.mixture_density)), axis=0) + 0.001)
        self.log_mixture_density_long = self.log_mixture_density_wide.reshape((self.K * self.n_users, 1))

        self.pos_i = self.assign_cluster(self.i, self.j_pos)
        self.neg_i = self.assign_cluster(self.i, self.j_neg)

        # for those samples that do not appear in the triplet, set the count to -1
        minus1 = T.zeros((self.n_users * self.K, 1), theano.config.floatX) - 1
        self.unique_i = theano.tensor.extra_ops.Unique()(T.concatenate((self.pos_i, self.neg_i)))

        one = numpy.asarray([1.0], dtype="float32").reshape((1, 1))

        # for those samples that do appear in the triplet, set the count to 0 first
        self.user_sample_counts_with_minus = T.inc_subtensor(minus1[self.unique_i, :], one)
        # add counts
        self.user_sample_counts_with_minus = T.inc_subtensor(self.user_sample_counts_with_minus[self.pos_i, :], one)
        self.user_sample_counts_with_minus = T.inc_subtensor(self.user_sample_counts_with_minus[self.neg_i, :], one)
        # add counts
        self.user_sample_counts =  T.inc_subtensor(self.user_sample_counts_with_minus[self.unique_i, :], -one) + 1

    def censor_updates(self):
        return super(KBPRModel, self).censor_updates() + [[self.V, self.max_norm], [self.U, self.max_norm]]

    def updates(self):
        updates = super(KBPRModel, self).updates()

        def fix(x):
            return T.minimum(1, T.maximum(0, x))
        if self.update_density and self.K > 1:
            updates += [[self.mixture_density, self.per_user_sample, fix]]
        if self.update_mu:
            updates += [[self.mixture_variance, self.user_sample_counts_with_minus.reshape((self.K, self.n_users))]]
        return updates

    def l2_reg(self):
        reg = super(KBPRModel, self).l2_reg()
        reg += [[(self.mixture_variance - self.variance_mu), self.lambda_variance, self.user_sample_counts.reshape((self.K, self.n_users))]]
        reg += [[self.cov_penalty(T.concatenate((self.V[self.unique_j], self.U[self.unique_i]))), self.lambda_cov, 1.0]]
        if self.K > 1:
            reg += [[(self.mixture_density - (1.0 / self.K)), self.lambda_density, self.per_user_sample]]
        for i in range(self.K):
            for j in range(i + 1, self.K):
                reg += [[self.U_norm_wide[i] - self.U_norm_wide[j],
                         self.lambda_mean_distance, self.per_user_sample]]
        return reg

    def cov_penalty(self, X):
        X = X - (X.sum(axis=0) / T.cast(X.shape[0], theano.config.floatX))
        return T.fill_diagonal(T.dot(X.T, X) / T.cast(X.shape[0], theano.config.floatX) , T.cast(0, theano.config.floatX))


    def assign_cluster(self, i, j):
        distance = ((self.U_norm_wide[:, i, :] - self.V_norm[j]) ** 2).sum(axis=2)
        normal = -(distance / (2 * (self.mixture_variance[:, i] ** 2))) - (
            T.log(self.mixture_variance[:, i] ** 2) / 2)
        return (normal + self.log_mixture_density_wide[:, i]).argmax(
            axis=0) * self.n_users + i

    def factor_delta(self):
        pos_j = self.pos_i
        neg_j = self.neg_i
        pos_distance = ((self.U_norm[pos_j, :] - self.V_norm[self.j_pos]) ** 2).sum(axis=1)
        pos_distance_p = -(pos_distance / (2 * (self.mixture_variance_long[pos_j, 0] ** 2))) - (
            T.log(self.mixture_variance_long[pos_j, 0] ** 2) / 2) + self.log_mixture_density_long[pos_j, 0]

        neg_distance = ((self.U_norm[neg_j, :] - self.V_norm[self.j_neg]) ** 2).sum(axis=1)
        neg_distance_p = -(neg_distance / (2 * (self.mixture_variance_long[neg_j, 0] ** 2))) - (
            T.log(self.mixture_variance_long[neg_j, 0] ** 2) / 2) + self.log_mixture_density_long[neg_j, 0]

        return pos_distance_p - neg_distance_p

    def bias_delta(self):
        if self.use_bias:
            return T.log(self.b[self.j_pos, 0]) - T.log(self.b[self.j_neg, 0])
        return 0

    def bias_score(self):
        if self.use_bias:
            return T.log(self.b).reshape((self.n_items,))
        return 0

    def factor_score(self):
        variance = self.mixture_variance.reshape((1, self.K, self.n_users))
        portion = self.log_mixture_density_wide.reshape((1, self.K, self.n_users))
        v_norm_wide = self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))  # (items, 1, ,1, factors)
        distance = ((self.U_norm_wide[:, self.i, :] - v_norm_wide) ** 2).sum(axis=3)  # (items, K, users)
        normal = -(distance / (2 * (variance[:, :, self.i] ** 2))) - (T.log(variance[:, :, self.i] ** 2) / 2)
        return (normal + portion[:, :, self.i]).max(axis=1).T  # transpose (items, users) to (users, items)

    def kmean(self, train_dict, K, learning_rate=1, normalize=True):
        # variables
        clusters = theano.shared(
            numpy.random.normal(0, 1 / (self.n_factors ** 0.5), (self.n_users * K, self.n_factors)).astype(
                theano.config.floatX))
        if normalize:
            clusters_norm = self.normalize(clusters)
        else:
            clusters_norm = clusters
        gradient_history = theano.shared(
            numpy.zeros((self.n_users * K, self.n_factors)).astype(theano.config.floatX))
        prev_cost = theano.shared(numpy.cast['float32'](numpy.Infinity))

        # positive samples
        tuples = []
        sample_counts = []
        for u, items in train_dict.items():
            for i in items:
                tuples.append((u, i, 0))
                sample_counts.append(float(len(items)))
        sample_counts = numpy.asarray(sample_counts, theano.config.floatX)

        # theano function
        clusters_norm_wide = clusters_norm.reshape((K, self.n_users, self.n_factors))
        distance = ((clusters_norm_wide[:, self.i, :] - self.V_norm[self.j_pos]) ** 2).sum(axis=2)
        assign = distance.argmin(axis=0) * self.n_users + self.i
        cost = (((clusters_norm[assign] - self.V_norm[self.j_pos]) ** 2).sum(axis=1) / sample_counts).sum()

        # adagrad
        gradient = T.grad(cost, wrt=clusters)
        new_history = gradient_history + (gradient ** float(2))
        adjusted_grad = gradient / ((new_history ** float(0.5)) + float(1e-10))
        f = theano.function([self.triplet], [cost, (prev_cost - cost) / prev_cost], updates=[
            [prev_cost, cost],
            [gradient_history, new_history],
            [clusters, clusters - (adjusted_grad * float(learning_rate))]])
        i = 0
        while True:
            i += 1
            cur_cost, diff = f(tuples)
            print ("Iter %d, Cost %g, Converge %g" % (i, cur_cost, diff))
            if abs(diff) < 1e-6:
                break
        return clusters.get_value()

    def loss(self, delta):
        ret = T.maximum(0, self.margin - delta)
        if self.use_warp:
            log_rank = T.log(float(self.n_items) / (self.trials[self.triplet[:, 3], 0] + 1)) / T.log(float(self.n_items))
            ret *= T.maximum(1E-3, log_rank)

        return ret

    def sample_uneven(self, train_dict):
        users, items = dict_to_coo(train_dict, self.n_users, self.n_items).nonzero()
        triplet = numpy.zeros((len(users), 4), "int64")

        triplet[:,0] = users
        triplet[:,1] = items
        triplet[:,3] = numpy.arange(len(users))


        while True:
            triplet[:,2] = numpy.random.randint(0, self.n_items, size=len(users))
            for i in xrange(len(users)):
                while triplet[i, 2] in train_dict[triplet[i, 0]]:
                    triplet[i, 2] = numpy.random.randint(0, self.n_items)
            yield triplet

    def train_hard(self, triplet, adagrad_val):
        total_hard = 0
        hard_case_loc = []

        for index, (u, pos, neg, id) in enumerate(triplet):

            hard_cases = self.hard_cases[u]
            if len(hard_cases) > 0:
                all_hard_cases = numpy.asarray(hard_cases.keys())
                if len(hard_cases) < self.per_user_sample:
                    triplet[index, 1:4] = all_hard_cases
                    hard_case_loc.extend(range(base, base + len(hard_cases)))
                    total_hard += len(hard_cases)
                else:
                    samples_index = numpy.random.choice(range(len(hard_cases)), replace=False,
                                                        size=self.per_user_sample)
                    triplet[base:base + self.per_user_sample, 1:4] = all_hard_cases[samples_index]
                    hard_case_loc.extend(range(base, base + self.per_user_sample))
                    total_hard += self.per_user_sample

        print "Hard Sample Cases:" + str(total_hard)
        loss, per_sample_losses = self.f(triplet, adagrad_val)
        per_sample_losses_real = -(per_sample_losses - self.margin)

        threshold = numpy.percentile(per_sample_losses_real[per_sample_losses_real < self.hard_case_margin], 20)
        # add hard case
        hard_triplet = triplet[(per_sample_losses_real < self.hard_case_margin) & (per_sample_losses_real > threshold)]
        for u, pos, neg, id in hard_triplet:
            case = (pos, neg, id)
            hard_cases = self.hard_cases[u]
            if case in hard_cases:
                hard_cases[case] += 1
                if hard_cases[case] > self.hard_case_chance:
                    del hard_cases[case]
            else:
                hard_cases[case] = 1
        # remove cases that are not hard any more
        for loc in hard_case_loc:
            if per_sample_losses_real[loc] > self.hard_case_margin or per_sample_losses_real[loc] < threshold:
                u, pos, neg, id = triplet[loc]
                hard_cases = self.hard_cases[u]
                del hard_cases[(pos, neg, id)]

        return loss, per_sample_losses

    def gen_updates(self, cost, per_sample_losses):
        all_cases = self.triplet[:, 3]
        violated_cases = self.triplet[per_sample_losses > 0, 3]

        new_trials = T.inc_subtensor(self.trials[all_cases, :], 1.0)
        new_trials = T.set_subtensor(new_trials[violated_cases, :], 0)

        return super(KBPRModel, self).gen_updates(cost, per_sample_losses) + \
               [[self.trials, new_trials]]

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False, profile=False):
        if self.train_dict != train_dict:
            self.train_dict = train_dict

            if self.uneven_sample:
                self.sample_generator = self.sample_uneven(train_dict)
            else:
                self.sample_generator = self.sample(train_dict)

            self.f = None
            self.pos_sample_count = reduce(lambda a, items: a + len(items), train_dict.values(), 0)
            self.trials = theano.shared(numpy.zeros((self.pos_sample_count, 1)), theano.config.floatX)

        if self.f is None:
            cost, per_sample_losses = self.cost()
            self.f = theano.function(
                inputs=[self.triplet, self.adagrad, self.unique_j],
                outputs=[cost, per_sample_losses],
                updates=self.gen_updates(cost, per_sample_losses),
                profile=profile
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
        for triplet, unique_i, unique_j in self.sample_generator:
            sample_time += time.time() - sample_start
            # update latent vectors
            training_start = time.time()
            if hard_case:
                loss, per_sample_losses = self.train_hard(triplet,  adagrad_val)
            else:
                if self.batch_size > 0:
                    indices = numpy.split(numpy.random.permutation(len(triplet)), numpy.arange(self.batch_size, len(triplet), self.batch_size))
                    for index in indices:
                        if len(index) > 0:
                            loss, per_sample_losses = self.f(triplet[index], adagrad_val)
                else:
                    loss, per_sample_losses = self.f(triplet, adagrad_val, unique_j)
            losses.append(loss)
            print numpy.sum(per_sample_losses > 0) / float(len(per_sample_losses))

            training_time += time.time() - training_start
            epoch_index += 1
            if epoch_index == epoch:
                sys.stderr.write(
                    "Train Time: %g Sample Time: %g\n" % (training_time, sample_time))
                sys.stderr.flush()
                return numpy.mean(losses)
            sample_start = time.time()

    def wrong_hard_case_rate(self, valid_dict):
        fail = 0
        total = 0
        for u, cases in enumerate(self.hard_cases):
            if u in valid_dict:
                total += len([neg for _, neg, _ in cases.keys()])
                fail += len([neg for _, neg, _ in cases.keys() if neg in valid_dict[u]])
        return fail / float(total)

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
            ["warp", self.use_warp]
        ]

    def copy(self):
        z = super(KBPRModel, self).copy()
        z.update(
            {
                "normalization": self.normalization,
                "margin": self.margin,
                "user_warp": self.use_warp,
                "K": self.K,
                "lambda_mean_distance": self.lambda_mean_distance,
                "lambda_variance": self.lambda_variance,
                "lambda_density": self.lambda_density,
                "lambda_cov": self.lambda_cov,
                "variance_mu_mu": self.variance_mu,
                "mixture_density": self.mixture_density.get_value(),
                "mixture_variance": self.mixture_variance.get_value()
            }
        )
        return z

class MaxMF(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, K=2, margin=1, hard_case_margin=-1,
                 mixture_density=None,
                 use_bias=False,
                 use_warp=True,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_bias=0.1,
                 hard_case_chances=2,
                 learning_rate=0.01,
                 uneven_sample=False,
                 per_user_sample=10,
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
                                    use_warp=use_warp,
                                    lambda_u=lambda_u,
                                    lambda_v=lambda_v,
                                    lambda_bias=lambda_bias,
                                    lambda_mean_distance=0.0,
                                    lambda_variance=0.0,
                                    uneven_sample=uneven_sample,
                                    hard_case_chances=hard_case_chances,
                                    learning_rate=learning_rate,
                                    per_user_sample=per_user_sample,
                                    batch_size=batch_size)

    def assign_cluster(self, i, j):
        return (self.U_norm_wide[:, i, :] * self.V_norm[j]).sum(axis=2).argmax(axis=0) * self.n_users + i

    def factor_delta(self):
        pos_j = self.assign_cluster(self.i, self.j_pos)
        neg_j = self.assign_cluster(self.i, self.j_neg)
        return ((self.U_norm[pos_j, :] * self.V_norm[self.j_pos]) -
                (self.U_norm[neg_j, :] * self.V_norm[self.j_neg])).sum(axis=1)

    def bias_delta(self):
        if self.use_bias:
            return self.b[self.j_pos, 0] - self.b[self.j_neg, 0]
        return 0

    def bias_score(self):
        if self.use_bias:
            return self.b.reshape((self.n_items,))
        return 0

    def factor_score(self):
        # (items, K, users)
        scores = (self.U_norm_wide[:, self.i, :] * self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))).sum(axis=3)
        return scores.max(axis=1).T


class KNormalBPRModel(KBPRModel):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, K=2, margin=1, U_mixture_portion=None,
                 lambda_u=0.1,
                 lambda_mean_distance=10.0,
                 lambda_variance=1.0,
                 lambda_density=1.0,
                 variance_mu=1.0,
                 update_mu=True,
                 learning_rate=0.01,
                 per_user_sample=10):

        super(KNormalBPRModel, self).__init__(
            n_factors, n_users, n_items,
            U=U, V=V, K=K, margin=margin,
            U_mixture_portion=U_mixture_portion,
            lambda_u=lambda_u,
            lambda_mean_distance=lambda_mean_distance,
            lambda_variance=lambda_variance,
            lambda_density=lambda_density,
            variance_mu=variance_mu,
            update_mu=update_mu,
            learning_rate=learning_rate,
            per_user_sample=per_user_sample)

    def prob(self, i, j, k):
        i = k * self.n_users + i
        log_density = self.log_mixture_density_long[i, 0]
        mu = (self.mixture_variance_long[i, 0])
        norm = T.exp(-((self.U_norm[i, :] - self.V_norm[j, :]) ** 2).sum(axis=1) / (2 * (mu ** 2)) + log_density)
        norm /= mu
        return norm

    def factor_score(self):
        mu = self.mixture_variance.reshape((1, self.K, self.n_users))[:, :, self.i]
        log_density = self.log_mixture_density_wide.reshape((1, self.K, self.n_users))[:, :, self.i]
        V = self.V_norm.reshape((self.n_items, 1, 1, self.n_factors))  # (items, 1, ,1, factors)
        U = self.U_norm_wide[:, self.i, :]  # (K, users, factors)
        distance = ((U - V) ** 2).sum(axis=3)  # (items, K, users)
        return (T.exp(-distance / (2 * (mu ** 2)) + log_density) / mu).sum(axis=1).T  # (user, items)

    def delta(self):
        pos_distance = self.prob(self.i, self.j_pos, 0)
        for i in range(1, self.K):
            pos_distance += self.prob(self.i, self.j_pos, i)
        neg_distance = self.prob(self.i, self.j_neg, 0)
        for i in range(1, self.K):
            neg_distance += self.prob(self.i, self.j_neg, i)
        return T.log(pos_distance) - T.log(neg_distance)


import multiprocessing
class LightFMModel(Model):
    def __init__(self, n_factors, n_users, n_items, lambda_u=0.0, lambda_v=0.0, learning_rate=0.01, loss="warp",
                 use_bias=True):

        super(LightFMModel, self).__init__(n_users, n_items)
        self.model = LightFM(learning_rate=learning_rate, loss=loss, no_components=n_factors, item_alpha=lambda_u,
                             user_alpha=lambda_v)
        self.train_coo = None
        self.use_bias = use_bias

    def train(self, train_dict, epoch=1, adagrad=False, hard_case=False):
        if self.train_coo is None:
            self.train_coo = dict_to_coo(train_dict, self.n_users, self.n_items)

        return self.model.fit_partial(self.train_coo, epochs=epoch, verbose=True,
                                      num_threads=multiprocessing.cpu_count())

    def validate(self, train_dict, valid_dict, per_user_sample=100):
        return self.auc(valid_dict, train_dict)

    def scores_for_users(self, users):
        for u in users:
            yield self.model.predict(u, numpy.arange(self.n_items), num_threads=multiprocessing.cpu_count())
