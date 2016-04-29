import numpy
from collections import defaultdict
import theano
import theano.tensor as T
import random
import time
from theano.ifelse import ifelse
import copy
import sys


def recall(ranks, likes_dict, exclude_dict, top_n):
    recalls = []
    for i in range(len(ranks)):
        if len(likes_dict[i]) > 0:
            user_ranks = ranks[i]
            index = 0
            hit = 0
            for item in user_ranks:
                if i in exclude_dict and item in exclude_dict[i]:
                    continue
                elif item in likes_dict[i]:
                    hit += 1
                index += 1
                if index == top_n:
                    break
            recalls.append(hit * 1.0 / len(likes_dict[i]))
    return numpy.mean(recalls)


def auc(ranks, likes_dict, exclude_dict):
    t = time.time()
    aucs = []
    all_count = len(ranks[0])
    for i in range(len(ranks)):
        excludes = []
        if i in exclude_dict[i]:
            excludes = exclude_dict[i]
        likes = likes_dict[i]
        pos_count = len(likes)
        neg_count = all_count - pos_count - len(excludes)
        total = pos_count * neg_count
        hit = 0
        rest_excludes = len(excludes)
        if len(likes) > 0:
            for j in ranks[i]:
                if rest_excludes > 0 and j in excludes:
                    rest_excludes -= 1
                elif j in likes:
                    # pos
                    hit += neg_count
                    pos_count -= 1
                    if pos_count == 0:
                        break
                else:
                    neg_count -= 1
            aucs.append(hit * 1.0 / total)
    print (time.time() - t)
    return numpy.mean(aucs)


class Model(object):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, lambda_u=0.01, lambda_v=0.01):
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_v = lambda_v
        self.lambda_u = lambda_u
        self.V = V
        self.U = U

        if U is None:
            self.U = theano.shared(
                value=numpy.random.normal(0, 0.1, (n_users, n_factors)).astype(theano.config.floatX),
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
                value=numpy.random.normal(0, 0.1, (n_items, n_factors)).astype(theano.config.floatX),
                name='V',
                borrow=True
            )

        else:
            self.V = theano.shared(
                value=V.astype(theano.config.floatX),
                name='V',
                borrow=True
            )

    def ranks(self):
        return numpy.zeros((self.n_users, self.n_items))

    def auc(self, train_dict, valid_dict):
        ranks = self.ranks()
        return auc(ranks, valid_dict, train_dict)

    def eval(self, train_dict, valid_dict):
        ranks = self.ranks()
        return {"train_recall@10": recall(ranks, train_dict, defaultdict(set), 10),
                "valid_recall@10": recall(ranks, valid_dict, train_dict, 10),
                "train_recall@100": recall(ranks, train_dict, defaultdict(set), 100),
                "valid_recall@100": recall(ranks, valid_dict, train_dict, 100),
                "train_auc": auc(ranks, train_dict, defaultdict(set)),
                "valid_auc": auc(ranks, valid_dict, train_dict)}

    def copy(self):
        return {"n_factors": self.n_factors,
                "n_users": self.n_users,
                "n_items": self.n_items,
                "lambda_v": self.lambda_v,
                "lambda_u": self.lambda_u,
                "V": self.V.get_value(),
                "U": self.U.get_value()}


class BPRModel(Model):
    def __init__(self, n_factors, n_users, n_items, U=None, V=None, b=None, lambda_u=0.1, lambda_v=0.1, lambda_b=0.1,
                 learning_rate=0.05, per_user_sample=10):
        Model.__init__(self, n_factors, n_users, n_items, U, V, lambda_u, lambda_v)
        # cache train_dict
        self.train_dict = None
        # cache sample_gen
        self.sample_generator = None
        self.lambda_b = lambda_b
        self.b = b
        self.learning_rate = learning_rate
        self.per_user_sample = per_user_sample
        if b is None:
            self.b = theano.shared(
                value=numpy.random.normal(0, 0.1, self.n_items).astype(theano.config.floatX),
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
        self.i = T.lscalar('i')
        # a vector of pos item ids
        self.j_pos = T.lvector('j_pos')
        # a vector of neg item ids
        self.j_neg = T.lvector('j_neg')
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
            {"lambda_b": self.lambda_b,
             "b": self.b.get_value(),
             "learning_rate": self.learning_rate})
        return z

    def __repr__(self):
        return "BPR_n_factors %d n_users %d n_items %d lambda_u %f lambda_v %f lambda_b %f  lr %f per_user_sample %d" % \
               (self.n_factors, self.n_users, self.n_items, self.lambda_u, self.lambda_v, self.lambda_b,
                self.learning_rate, self.per_user_sample)

    def l2_reg(self):
        return [(self.u, self.lambda_u * self.per_user_sample),
                (self.v_pos, self.lambda_v),
                # allow negative sample to change with larger scale
                (self.v_neg, self.lambda_v / 10.0),
                (self.b_pos, self.lambda_b),
                (self.b_neg, self.lambda_b)
                ]

    def l1_reg(self):
        return []

    def global_updates(self, cost):
        return []

    def updates(self, cost):
        u_i_grad = T.grad(cost=cost, wrt=self.u) / self.per_user_sample
        v_pos_grad = T.grad(cost=cost, wrt=self.v_pos)
        v_neg_grad = T.grad(cost=cost, wrt=self.v_neg)
        b_pos_grad = T.grad(cost=cost, wrt=self.b_pos)
        b_neg_grad = T.grad(cost=cost, wrt=self.b_neg)

        return [[self.U, T.set_subtensor(self.U[self.i], self.U[self.i] - (u_i_grad * self.learning_rate))],
                [self.V, T.set_subtensor(
                    T.set_subtensor(self.V[self.j_pos], self.v_pos - v_pos_grad * self.learning_rate)[self.j_neg],
                    (self.v_neg - v_neg_grad * self.learning_rate)
                )],
                [self.b, T.set_subtensor(
                    T.set_subtensor(self.b[self.j_pos], self.b_pos - b_pos_grad * self.learning_rate)[self.j_neg],
                    (self.b_neg - b_neg_grad * self.learning_rate)
                )]
                ]

    def cache_updates(self):
        return []

    def delta(self, use_cache=True):
        # user vector
        delta = T.dot(self.u, (self.v_pos - self.v_neg).T) + self.b_pos - self.b_neg
        return delta

    def scores(self):
        return T.dot(self.U, self.V.T) + self.b

    def cost(self, use_cache=True):
        # user vector
        delta = self.delta(use_cache=use_cache)
        cost = -T.log(T.nnet.sigmoid(delta)).sum()
        for term, l in self.l2_reg():
            if l > 0.0:
                cost += (term ** 2).sum() * l
        for term, l in self.l1_reg():
            if l > 0.0:
                cost += (abs(term)).sum() * l
        return cost

    def sample(self, train_dict, exclude_dict=None):
        all_items = range(self.n_items)
        samples = defaultdict(list)
        while True:
            for u in numpy.random.choice(train_dict.keys(), len(train_dict.keys()), replace=False):
                pos = []
                neg = []
                if len(train_dict[u]) == 0:
                    continue
                train = train_dict[u]
                exclude = None
                if exclude_dict is not None and u in exclude_dict:
                    exclude = exclude_dict[u]
                for i in range(self.per_user_sample):
                    # pos samples deplete. reset
                    if len(samples[u]) == 0:
                        pos_samples = list(train)
                        random.shuffle(pos_samples)
                        samples[u] = pos_samples
                    pos_item = samples[u].pop()
                    neg_item = random.choice(all_items)
                    while neg_item in train or (exclude is not None and neg_item in exclude):
                        neg_item = random.choice(all_items)
                    pos.append(pos_item)
                    neg.append(neg_item)
                yield (u, pos, neg)

    def train(self, train_dict, epoch=1):

        if self.train_dict != train_dict:
            self.sample_generator = self.sample(train_dict)
        if self.f is None:
            # adopt the version of cost that use cache to speed up!
            cost = self.cost(use_cache=True)
            self.f = theano.function(
                inputs=[self.i, self.j_pos, self.j_neg],
                outputs=cost,
                updates=self.updates(cost)
            )

            # adopt the version of cost that does not use cache
            # (because the cached values are dependant on the global parameters which we are going to update)
            cost_use_no_cache = self.cost(use_cache=False)
            global_updates = self.global_updates(cost_use_no_cache)
            if global_updates is not None and len(global_updates) > 0:
                self.global_f = theano.function(
                    inputs=[self.i, self.j_pos, self.j_neg],
                    outputs=cost_use_no_cache,
                    updates=global_updates
                )
            cache_updates = self.cache_updates()
            if cache_updates is not None and len(cache_updates) > 0:
                self.cache_f = theano.function(
                    inputs=[],
                    outputs=None,
                    updates=cache_updates
                )

        # perform global parameters updates #
        if self.global_f is not None:
            num = 0
            for u, pos, neg in self.sample_generator:
                # update latent vectors
                if numpy.random.random() > 0.9:
                    self.global_f(u, pos, neg)
                num += 1
                if num == epoch * self.n_users:
                    break

        # compute cache #
        if self.cache_f is not None:
            self.cache_f()

        num = 0
        train_loss = 0.0
        # perform latent vector updates #
        for u, pos, neg in self.sample_generator:
            # update latent vectors

            train_loss += self.f(u, pos, neg)
            num += 1
            if num == epoch * self.n_users:
                break
        return train_loss / num

    def validate(self, train_dict, valid_dict, runs=1):
        if self.validate_f is None:
            # compute AUC
            delta = self.delta()
            self.validate_f = theano.function(
                inputs=[self.i, self.j_pos, self.j_neg],
                outputs=-T.switch(T.gt(delta, 0), 1.0, 0.0).sum() / self.per_user_sample
            )
        num = 0
        losses = 0.0
        # sample triplet with the sample generator
        for u, pos, neg in self.sample(valid_dict, train_dict):
            # update latent vectors
            losses += self.validate_f(u, pos, neg)
            num += 1
            if num == self.n_users * runs:
                break
        return losses / num

    def ranks(self):
        if self.score_f is None:
            self.score_f = theano.function(
                inputs=[],
                outputs=self.scores()
            )

        return numpy.argsort(-self.score_f())


class BPRModelWithVisualBias(BPRModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U=None, V=None, b=None, visual_b=None,
                 use_visual_bias=True,
                 lambda_u=0.1, lambda_v=0.1, lambda_b=0.1, lambda_visual_b=0.1, learning_rate=0.01,
                 visual_bias_learning_rate=0.001,
                 per_user_sample=10):
        BPRModel.__init__(self, n_factors, n_users, n_items, U, V, b, lambda_u, lambda_v, lambda_b, learning_rate,
                          per_user_sample)
        self.lambda_visual_b = lambda_visual_b
        self.visual_bias_learning_rate = visual_bias_learning_rate
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

        self.V_bias_cache = theano.shared(
            value=numpy.zeros(n_items).astype(theano.config.floatX),
            name='visual_bias_cache',
            borrow=True
        )

        # diff between two raw visual features
        self.visual_diff = self.V_features[self.j_pos] - self.V_features[self.j_neg]

    def __repr__(self):
        return "BPRWithVisualBias_n_factors %d n_users %d n_items %d lambda_u %f lambda_v %f lambda_b %f  " \
               "lambda_visual_b %f lr %f visual_bias_lr %f per_user_sample %d user_visual_bias %s" % \
               (self.n_factors, self.n_users, self.n_items, self.lambda_u, self.lambda_v,
                self.lambda_b, self.lambda_visual_b, self.learning_rate, self.visual_bias_learning_rate,
                self.per_user_sample, self.use_visual_bias)

    def copy(self):
        z = super(BPRModelWithVisualBias, self).copy()
        z.update(
            {"visual_b": self.visual_b.get_value(),
             "use_visual_bias": self.use_visual_bias,
             "lambda_visual_b": self.lambda_visual_b,
             "visual_bias_learning_rate": self.visual_bias_learning_rate
             }
        )
        return z

    def global_updates(self, cost):
        updates = super(BPRModelWithVisualBias, self).global_updates(cost)
        if self.use_visual_bias:
            visual_b_grad = T.grad(cost=cost, wrt=self.visual_b) / self.per_user_sample
            return updates + [
                # use a separate learning rate for embedding weights. It is usually much smaller
                [self.visual_b, self.visual_b - (visual_b_grad * self.visual_bias_learning_rate)]
            ]
        else:
            return updates

    def cache_updates(self):
        if self.use_visual_bias:
            return [[self.V_bias_cache, T.dot(self.V_features, self.visual_b)]]
        else:
            return []

    def delta(self, use_cache=True):
        delta = super(BPRModelWithVisualBias, self).delta(use_cache)
        if self.use_visual_bias:
            if use_cache:
                delta += self.V_bias_cache[self.j_pos] - self.V_bias_cache[self.j_neg]
            else:
                delta += T.dot(self.visual_diff, self.visual_b)
        return delta

    def scores(self):
        return super(BPRModelWithVisualBias, self).scores() + T.dot(self.V_features, self.visual_b).T

    def l2_reg(self):
        return BPRModel.l2_reg(self) + \
               [[self.visual_b, self.lambda_visual_b * self.per_user_sample]]

    def l1_reg(self):
        return BPRModel.l1_reg(self) + \
               [[self.visual_b, self.lambda_visual_b * self.per_user_sample]]


class VisualBPRAbstractModel(BPRModelWithVisualBias):
    def __init__(self, n_factors, n_users, n_items, n_embedding_dim, V_features, U=None, V=None, b=None, visual_b=None,
                 weights=None, use_visual_bias=True,
                 lambda_u=0.1, lambda_v=0.1, lambda_b=0.1, lambda_visual_b=0.1, lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01, visual_bias_learning_rate=0.001,
                 embedding_learning_rate=0.0001, learning_rate=0.01, per_user_sample=10):
        BPRModelWithVisualBias.__init__(self, n_factors, n_users, n_items, V_features, U, V, b, visual_b,
                                        use_visual_bias, lambda_u, lambda_v, lambda_b, lambda_visual_b, learning_rate,
                                        visual_bias_learning_rate,
                                        per_user_sample)

        self.lambda_weight_l1 = lambda_weight_l1
        self.lambda_weight_l2 = lambda_weight_l2
        self.n_embedding_dim = n_embedding_dim
        # create weights
        raw_visual_feature_factors = len(self.V_features.get_value()[0])
        if weights is None:
            self.weights = theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                    high=numpy.sqrt(6. / (raw_visual_feature_factors + n_embedding_dim)),
                    size=(raw_visual_feature_factors, n_embedding_dim)
                ),
                dtype=theano.config.floatX
            ), borrow=True, name='weights')
        else:
            self.weights = theano.shared(numpy.asarray(
                weights,
                dtype=theano.config.floatX
            ), borrow=True, name='weights')

        # learning rate
        self.embedding_learning_rate = embedding_learning_rate

        self.V_embedding_cache = theano.shared(
            value=numpy.zeros((n_items, n_embedding_dim)).astype(theano.config.floatX),
            name='V_embedding_cache',
            borrow=True
        )

    def copy(self):
        z = super(VisualBPRAbstractModel, self).copy()
        z.update(
            {"lambda_weight_l1": self.lambda_weight_l1,
             "lambda_weight_l2": self.lambda_weight_l2,
             "weights": self.weights.get_value(),
             "n_embedding_dim": self.n_embedding_dim,
             "embedding_learning_rate": self.embedding_learning_rate}
        )
        return z

    def global_updates(self, cost):
        updates = super(VisualBPRAbstractModel, self).global_updates(cost)
        weight_grad = T.grad(cost=cost, wrt=self.weights)
        return updates + [
            # use a separate learning rate for embedding weights. It is usually much smaller
            [self.weights, self.weights - (weight_grad * self.embedding_learning_rate)]
        ]

    def cache_updates(self):
        updates = super(VisualBPRAbstractModel, self).cache_updates()
        updates += [[self.V_embedding_cache, T.dot(self.V_features, self.weights)]]
        return updates
    def l2_reg(self):
        return super(BPRModelWithVisualBias, self).l2_reg() + [[self.weights, self.lambda_weight_l2]]

    def l1_reg(self):
        return super(BPRModelWithVisualBias, self).l1_reg() + [[self.weights, self.lambda_weight_l1]]



class VisualBPRConcatModel(VisualBPRAbstractModel):
    def __init__(self, n_factors, n_visual_factors, n_users, n_items, V_features, U=None, V=None, U_visual=None, b=None,
                 visual_b=None,
                 weights=None, use_visual_bias=True,
                 lambda_u=0.1, lambda_v=0.1, lambda_b=0.1, lambda_visual_b=0.1, lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 embedding_learning_rate=0.0001, learning_rate=0.01, visual_bias_learning_rate=0.001,
                 per_user_sample=10):

        VisualBPRAbstractModel.__init__(self, n_factors, n_users, n_items, n_visual_factors, V_features, U, V, b,
                                        visual_b, weights=weights, use_visual_bias=use_visual_bias,
                                        lambda_u=lambda_u, lambda_v=lambda_v, lambda_b=lambda_b,
                                        lambda_visual_b=lambda_visual_b, lambda_weight_l2=lambda_weight_l2,
                                        lambda_weight_l1=lambda_weight_l1,
                                        visual_bias_learning_rate=visual_bias_learning_rate,
                                        embedding_learning_rate=embedding_learning_rate, learning_rate=learning_rate,
                                        per_user_sample=per_user_sample)

        self.n_visual_factors = n_visual_factors
        self.U_visual = U_visual
        # create user visual factor matrix
        if U_visual is None:
            self.U_visual = theano.shared(
                value=numpy.random.normal(0, 0.1, (n_users, n_visual_factors)).astype(theano.config.floatX),
                name='U_visual',
                borrow=True
            )
        else:
            self.U_visual = theano.shared(
                value=U_visual.astype(theano.config.floatX),
                name='U_visual',
                borrow=True
            )
        # user visual vector
        self.u_visual = self.U_visual[self.i]

    def __repr__(self):
        return "VisualBPRConcat_n_factors %d n_visual_factors %d n_users %d n_items %d lambda_u %f " \
               "lambda_v %f lambda_b %f lambda_visual_b %f lambda_w_l1 %f lambda_w_l2 %f  lr %f embedding_learning_rate %f " \
               "per_user_sample %d use_visual_bias %s" % \
               (self.n_factors, self.n_visual_factors, self.n_users, self.n_items, self.lambda_u, self.lambda_v,
                self.lambda_b, self.lambda_visual_b,
                self.lambda_weight_l1, self.lambda_weight_l2,
                self.learning_rate, self.embedding_learning_rate, self.per_user_sample, self.use_visual_bias)

    def copy(self):
        z = super(VisualBPRConcatModel, self).copy()
        z.update(
            {"U_visual": self.U_visual.get_value()}
        )
        return z

    def updates(self, cost):
        updates = super(VisualBPRConcatModel, self).updates(cost)
        u_visual_grad = T.grad(cost=cost, wrt=self.u_visual) / self.per_user_sample
        return updates + [
            [self.U_visual,
             T.set_subtensor(self.U_visual[self.i], self.u_visual - (u_visual_grad * self.learning_rate))]
        ]

    def delta(self, use_cache=True):
        # compute difference of visual features in the embedding
        delta = super(VisualBPRConcatModel, self).delta(use_cache=use_cache)
        if use_cache:
            embedding_diff = self.V_embedding_cache[self.j_pos] - self.V_embedding_cache[self.j_neg]
        else:
            # compute distance different to the user visual vector
            embedding_diff = T.dot(self.visual_diff, self.weights)

        delta += T.dot(self.u_visual, embedding_diff.T)
        return delta

    def scores(self):
        return super(VisualBPRConcatModel, self).scores() + T.dot(self.U_visual, T.dot(self.V_features, self.weights).T)

    def l2_reg(self):
        return super(VisualBPRConcatModel, self).l2_reg() + [[self.u_visual, self.lambda_u * self.per_user_sample]]


class VisualBPRStackModel(VisualBPRAbstractModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U=None, V=None, b=None, visual_b=None,
                 weights=None, use_visual_bias=True,
                 lambda_u=0.1, lambda_v=0.1, lambda_b=0.1, lambda_visual_b=0.1, lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01, visual_bias_learning_rate=0.001,
                 embedding_learning_rate=0.0001, learning_rate=0.01, per_user_sample=10):

        VisualBPRAbstractModel.__init__(self, n_factors, n_users, n_items, n_factors, V_features, U, V, b,
                                        visual_b, weights=weights, use_visual_bias=use_visual_bias,
                                        lambda_u=lambda_u, lambda_v=lambda_v, lambda_b=lambda_b,
                                        lambda_visual_b=lambda_visual_b, lambda_weight_l2=lambda_weight_l2,
                                        lambda_weight_l1=lambda_weight_l1,
                                        visual_bias_learning_rate=visual_bias_learning_rate,
                                        embedding_learning_rate=embedding_learning_rate, learning_rate=learning_rate,
                                        per_user_sample=per_user_sample)

    def __repr__(self):
        return "VisualBPRStack_n_factors %d  n_users %d n_items %d lambda_u %f " \
               "lambda_v %f lambda_b %f lambda_visual_b %f lambda_w_l1 %f lambda_w_l2 %f  lr %f " \
               "embedding_learning_rate %f " \
               "per_user_sample %d use_visual_bias %s" % \
               (self.n_factors, self.n_users, self.n_items, self.lambda_u, self.lambda_v,
                self.lambda_b, self.lambda_visual_b,
                self.lambda_weight_l1, self.lambda_weight_l2,
                self.learning_rate, self.embedding_learning_rate, self.per_user_sample, self.use_visual_bias)

    def delta(self, use_cache=True):
        delta = super(VisualBPRStackModel, self).delta(use_cache=use_cache)
        if use_cache:
            embedding_diff = (self.V_embedding_cache[self.j_pos] - self.V_embedding_cache[self.j_neg])
        else:
            embedding_diff = T.dot(self.visual_diff, self.weights)
        delta += T.dot(self.u, embedding_diff.T)
        return delta

    def scores(self):
        return super(VisualBPRStackModel, self).scores() + T.dot(self.U, T.dot(self.V_features, self.weights).T)


class UserVisualModel(VisualBPRStackModel):
    def __init__(self, n_factors, n_users, n_items, V_features, U=None, weights=None,
                 lambda_weight_l2=0.01,
                 lambda_weight_l1=0.01,
                 lambda_u=0.1, learning_rate=0.01, embedding_learning_rate=0.001,
                 per_user_sample=10):

        V = numpy.zeros((n_items, n_factors))
        b = numpy.zeros(n_items)

        VisualBPRAbstractModel.__init__(self, n_factors, n_users, n_items, n_factors, V_features, U, V, b,
                                        None, weights=weights, use_visual_bias=False,
                                        lambda_u=lambda_u, lambda_v=0, lambda_b=0,
                                        lambda_visual_b=0, lambda_weight_l2=lambda_weight_l2,
                                        lambda_weight_l1=lambda_weight_l1,
                                        visual_bias_learning_rate=0,
                                        embedding_learning_rate=embedding_learning_rate, learning_rate=learning_rate,
                                        per_user_sample=per_user_sample)

    def __repr__(self):
        return "UserVisualModel_n_factors %d n_users %d n_items %d lambda_u %f " \
               "lr %f per_user_sample %d" % \
               (self.n_factors, self.n_users, self.n_items, self.lambda_u, self.learning_rate,
                self.per_user_sample)

    def updates(self, cost):
        # only update U. do not update V
        u_i_grad = T.grad(cost=cost, wrt=self.u) / self.per_user_sample
        return [[self.U, T.set_subtensor(self.U[self.i], self.U[self.i] - (u_i_grad * self.learning_rate))]]

    def delta(self, use_cache=True):
        # only use the difference between U*visual embedding
        if use_cache:
            embedding_diff = self.V_embedding_cache[self.j_pos] - self.V_embedding_cache[self.j_neg]
        else:
            embedding_diff = T.dot(self.visual_diff, self.weights)
        delta = T.dot(self.u, embedding_diff.T)
        return delta


def early_stopping(model, train_dict, valid_dict, best_params_ret=None, n_epochs=1000,
                   validation_frequency=10, patience=200, **kwargs):
    sys.stderr.write("Model %s n_epochs %d\n" % (model, n_epochs))
    sys.stderr.flush()

    # early-stopping parameters
    # look as this many epochs
    patience_increase = 200  # wait this much longer when a new best is  found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    # go through this many
    # epochs before checking the model

    best_params = None
    best_validation_loss = numpy.inf
    best_train_loss = numpy.inf
    patience_for_train_loss = 2
    done_looping = False

    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        # Report "1" for first epoch, "n_epochs" for last epoch

        start = time.time()
        cur_train_loss = model.train(train_dict, epoch=validation_frequency, **kwargs)
        this_validation_loss = model.validate(train_dict, valid_dict)
        train_auc = model.validate(None, train_dict)
        epoch += validation_frequency
        sys.stderr.write(
            "\rEpoch: %d time: %d best: %f cur %f" % (epoch, time.time() - start, best_train_loss, cur_train_loss))
        sys.stderr.flush()

        # check train loss
        if cur_train_loss > best_train_loss:
            if patience_for_train_loss == 0:
                sys.stderr.write(
                    "\nEpoch: %d Train Loss Increase! best: %f cur %f\n" % (epoch, best_train_loss, cur_train_loss))
                sys.stderr.flush()
            else:
                patience_for_train_loss -= 1
        else:
            best_train_loss = cur_train_loss
            patience_for_train_loss = 2

        # check validation loss
        if this_validation_loss < best_validation_loss:
            # improve patience if loss improvement is good enough
            if this_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, epoch + patience_increase)
            best_params = model.copy()
            best_params["model"] = str(model)
            if best_params_ret is not None:
                if len(best_params_ret) == 0:
                    best_params_ret.append(best_params)
                best_params_ret[0] = best_params
            best_validation_loss = this_validation_loss

            sys.stderr.write("\nEpoch: %d New best valid loss(auc) %f train loss %f train auc %f patience %d\n" % (
                epoch, best_validation_loss, cur_train_loss, train_auc, patience))
            sys.stderr.flush()
        else:
            sys.stderr.write(
                "\nEpoch: %d Overfit!!?? valid loss(auc) %f best valid loss(auc) %f train loss %f train auc %f patience %d\n" % (
                    epoch, this_validation_loss, best_validation_loss, cur_train_loss, train_auc, patience))
            sys.stderr.flush()

        if patience <= epoch:
            done_looping = True
            break

    return best_params
