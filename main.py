import model
import utils
reload(model)
reload(utils)
import model
import utils
from model import *
from utils import *
from sklearn.manifold import TSNE
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pyprind


n_epochs = 2000
sparsity = 8
user_dict, photo_ids, user_ids = pickle.load(open("/home/ubuntu/dataset/dat_100_100000.p", "rb"))
# V_features = v_features(photo_ids)
V_features = numpy.load("/mnt/dat_100_100000.npy")
numpy.random.seed(1)
user_dict_reduced = dict(
    [(i_index, set(numpy.random.choice(list(items), size=min(len(items), 100), replace=False))) for i_index, items in
     user_dict.iteritems()])
n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict = preprocess(user_dict_reduced, portion=[8,1,1,0])
V, clusters, clusters_3 = pickle.load( open("tmp.p", "rb"))
kbpr1_c = pickle.load( open("kbpr1.p", "rb"))




# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmax, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

def run(model, **kwargs):
    early_stopping(model, train_dict, valid_dict, exclude_dict, pre="S-" + str(sparsity) + " ",
                   valid_per_user_sample=50, **kwargs)


def extract_kmean(m, train_dict, n):
    bar = pyprind.ProgBar(m.n_users)  # 1) initialization with number of iterations
    scores = [0] * m.n_users
    clusters = numpy.zeros((m.n_users, n, m.n_factors)).astype(theano.config.floatX)
    for i in range(n):
        clusters[:, i, :] = m.U.get_value()

    for u in xrange(m.n_users):
        bar.update()
        if u in train_dict and len(train_dict[u]) > 3:
            X = m.V.get_value()[list(train_dict[u])]
            clusterer = KMeans(n_clusters=n, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            clusters[u, :, :] = clusterer.cluster_centers_
            scores[u] = silhouette_score(X, cluster_labels)
    return scores, clusters

def plot_sse(m, train_dict):
    bar = pyprind.ProgBar(m.n_users)  # 1) initialization with number of iterations
    sse = []
    for u in xrange(m.n_users):
        bar.update()
        if len(train_dict[u]) > 2:
            X = m.V.get_value()[list(train_dict[u])]
            clusterer = KMeans(n_clusters=2, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            sse.append(numpy.sum((X - U[u])**2) / numpy.sum((X  - clusterer.cluster_centers_[cluster_labels]) ** 2))
    return scores, clusters



scores, clusters = extract_kmean(kbpr1, train_dict)


# 1 mixture

movie1 = KBPRModel(50, n_users, n_items,
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=1, margin=1, variance_mu=1, update_mu=False)
run(movie1, start_adagrad=100, start_hard_case=100)


# 2 mixtures
U = clusters
portions = numpy.zeros((n_users,1)).astype(theano.config.floatX) + 0.5
U_k =  numpy.ndarray.transpose(U, (1,0,2)).reshape((n_users*2,50))

kbpr2_kmean = KBPRModel(50, n_users, n_items, U=U_k, V=V, U_mixture_portion=portions.reshape(1, n_users),
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=2, margin=1, variance_mu=1)
run(kbpr2_kmean, start_adagrad=0, start_hard_case=0)



# 3 mixtures
U = clusters_3
portions = numpy.zeros((n_users,2)).astype(theano.config.floatX) + (1/3.0)
U_k =  numpy.ndarray.transpose(U, (1,0,2)).reshape((n_users*3,50))


kbpr3_kmean = KBPRModel(50, n_users, n_items, U=U_k, V=V, U_mixture_portion=portions.reshape(2, n_users),
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=3, margin = 1, variance_mu=1)



while True:
    run(kbpr3_kmean, start_adagrad=0, start_hard_case=0, n_epochs=10)
    kbpr3_kmean.recall(test_dict, exclude_dict, n_users=1000)
    print numpy.mean(kbpr3_kmean.mixture_variance.get_value())
    print numpy.mean((kbpr3_kmean.mixture_density.get_value() - 0.3333) ** 2)

run(kbpr3_kmean, start_adagrad=0, start_hard_case=0)



U = clusters_3
portions = numpy.zeros((n_users,2)).astype(theano.config.floatX) + (1/3.0)
U_k =  numpy.ndarray.transpose(U, (1,0,2)).reshape((n_users*3,50))

kbpr3_norm = KNormalBPRModel(50, n_users, n_items, U=U_k, V=V, U_mixture_portion=portions.reshape(2, n_users),
                             per_user_sample=50, learning_rate=0.01, lambda_u=1, K=3, margin=1.2, variance_mu=1,
                             lambda_density=10,
                             lambda_mean_distance=0,
                             lambda_variance=10)

#run(kbpr3_norm, start_adagrad=0, start_hard_case=0)

while True:
    run(kbpr3_norm, start_adagrad=0, start_hard_case=0, n_epochs=10)
    kbpr3_norm.recall(valid_dict, train_dict, n_users=1000)
    print numpy.mean(kbpr3_norm.mixture_variance.get_value())
    print numpy.mean((kbpr3_norm.mixture_density.get_value() - 0.3333) ** 2)


def stats(m, u):
    i_index = T.lscalar()
    items = T.lvector()
    valid_items = T.lvector()
    distances = ((m.U_norm_wide[:, i_index, :].reshape((2, 1, 50)) - m.V_norm) ** 2).sum(
        axis=2).T  # shape: (n_items, K)
    weighted_distances = -(distances / (2 * (m.mixture_variance[:, i_index] ** 2)))
    scores_K = (weighted_distances + m.U_mixture_portion_derived[:, i_index])
    assignments = scores_K.argmax(axis=1)
    scores = scores_K.max(axis=1)
    ranks = T.argsort(T.argsort(-scores))
    mixture_distance = T.sum((m.U_norm_wide[0, i_index, :] - m.U_norm_wide[1,
                                           i_index,
                                           :]) ** 2)
    avg_mixture_distance = T.sum((m.U_norm_wide[0, :, :] - m.U_norm_wide[1,
                                                             :,
                                                             :]) ** 2) / n_users
    my_variances = m.mixture_variance[:,u]
    mean_var = T.mean(m.mixture_variance)
    outputs = \
    [m.U_mixture_portion_derived[:, i_index],
     my_variances,
     mean_var,
     mixture_distance,
     avg_mixture_distance,
     T.sum(assignments) / float(n_items),
     T.sum(ranks[items]) / items.shape[0],
     T.sum(ranks[valid_items]) / valid_items.shape[0]]
    [portions, var, avg_var, distance, avg_mixture_distance, assignment, avg_rank, avg_valid] = theano.function([i_index, items, valid_items],
                                                                 outputs
                                                                 )(u, list(train_dict[u]), list(valid_dict[u]))
    print ("Portions %s,%s, Variance %s/%g, D %s/%s, Assign %g Rank %g/%g" %
           (portions, numpy.exp(portions), var, avg_var, distance, avg_mixture_distance, assignment, avg_rank, avg_valid))

while True:
    kbpr3_kmean.train(train_dict, adagrad=True, hard_case=True)
    kbpr3_kmean.recall(valid_dict, train_dict, n_users=1000)
    print "Valid " + str(kbpr3_kmean.validate(train_dict, valid_dict))

while True:
    kbpr1.train(train_dict, adagrad=True, hard_case=True)
    kbpr1.recall(valid_dict, train_dict, n_users=1000)
    print kbpr1.validate(train_dict, valid_dict)


kbpr3_kmean = KBPRModel(50, n_users, n_items, U=U_k, V=V, U_mixture_portion=portions.reshape(2, n_users),
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=3, margin = 1, variance_mu=1)
while True:
    run(kbpr3_kmean, start_adagrad=0, start_hard_case=0, n_epochs=10)
    kbpr3_kmean.recall(test_dict, exclude_dict)



for lambda_cov in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    lastfm_150_cov = KBPRModel(150, n_users_fm, n_items_fm, batch_size=100000,
                               per_user_sample=2000, learning_rate=0.1, lambda_u=0.01,
                               lambda_v=0.01, lambda_bias=0.01, use_bias=True,
                               K=1, margin=1, variance_mu=1, update_mu=True,
                               normalization=False, uneven_sample=True, lambda_variance=1, lambda_cov=lambda_cov)
    early_stopping(lastfm_150_cov, train_dict_fm, valid_dict_fm, test_dict_fm, pre="LastFM" + " ",
                   valid_per_user_sample=200, n_epochs=1000, start_adagrad=0000, start_hard_case=10000)

n_items_1m, n_users_1m, train_dict_1m, valid_dict_1m, test_dict_1m, exclude_dict_1m = preprocess(movielens1M(4), portion=[8,1,1,0])



n_items_fm, n_users_fm, train_dict_fm, valid_dict_fm, test_dict_fm, exclude_dict_fm = preprocess(lastfm(), portion=[8,1,1,0])

movie1 = KBPRModel(50, n_users_1m, n_items_1m,
                   per_user_sample=50, learning_rate=0.1, lambda_u=0.01, lambda_v=0.01, use_bias=True,
                   lambda_bias=0.01, K=1, margin=1, variance_mu=0.1, update_mu=False, hard_case_chances=5)

early_stopping(movie1, train_dict_1m, valid_dict_1m, test_dict_1m, pre="Movie ",
               valid_per_user_sample=50, start_adagrad=000, start_hard_case=1000, n_epochs=50)
while True:
    early_stopping(movie1, train_dict_1m, valid_dict_1m, test_dict_1m, n_epochs=20, pre="Movie ",
                   valid_per_user_sample=50, start_adagrad=0, start_hard_case=0)
    print movie1.precision(valid_dict_1m, train_dict_1m, n=10)[0]
    print movie1.wrong_hard_case_rate(valid_dict_1m)

U = clusters3
portions = numpy.zeros((n_users_1m,2)).astype(theano.config.floatX) + (1/3.0)
U_k =  numpy.ndarray.transpose(U, (1,0,2)).reshape((n_users_1m*3,50))
kbpr3 = KBPRModel(50, n_users_1m, n_items_1m, U=U_k, V=movie1.V.get_value(), U_mixture_portion=portions.reshape(2, n_users_1m),
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=3, margin=1, variance_mu=1)