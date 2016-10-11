from annoy import AnnoyIndex
from utils import *
from pymongo import MongoClient
import  pyprind

def eval_ours(id=-6260824840641569386, n=10, n_users=-1):
    model, train, valid, test, exclude = load_model(id)
    n_most_popular = 0
    users = valid.keys()
    bias = numpy.log(model.b.get_value())[:, 0]
    U = model.U.get_value()
    V = model.V.get_value()
    Var = model.mixture_variance.get_value()
    most_popular = numpy.argsort(bias)[0:n_most_popular]
    most_popular_vectors = V[most_popular]
    most_popular_bias = bias[most_popular]
    numpy.random.seed(1)
    if n_users > 0:
        sampled_users = numpy.random.choice(users, n_users)
    else:
        sampled_users = users
    for n_trees in [10, 50, 100, 200, 400]:
        t = AnnoyIndex(V.shape[1], metric="euclidean")  # Length of item vector that will be indexed
        # build trees
        for i in xrange(len(V)):
            t.add_item(i, V[i], bias=bias[i])
        t.build(n_trees)  # 10 trees
        print "Tree built: " + str(n_trees)

        for n_ann_samples in [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            print "N_Ann_Samples" + " " + str(n_ann_samples)
            recall = []
            bar = pyprind.ProgBar(len(sampled_users))
            total_time = 0.0
            for u in sampled_users:
                u_vector = U[u]
                variance = 2 * (Var[0, u] ** 2)
                likes = valid[u]
                to_exclude = train[u]
                if n_most_popular > 0:
                    most_popular_scores = (numpy.square(u_vector - most_popular_vectors)).sum(
                        1) / variance - most_popular_bias
                clock = time.clock()
                close_items, distances = t.get_nns_by_vector(u_vector, n + len(to_exclude),
                                                             rescale=1 / variance,
                                                             search_k=n_ann_samples, include_distances=True)
                total_time += time.clock() - clock
                distances = numpy.square(distances)
                scores = distances
                hits = 0
                seen = 0
                if n_most_popular > 0:
                    all_scores = numpy.append(scores, [most_popular_scores])
                else:
                    all_scores = scores

                for index in numpy.argsort(all_scores):
                    if index >= len(close_items):
                        j = most_popular[index - len(close_items)]
                    else:
                        j = close_items[index]
                    if j in to_exclude:
                        continue
                    else:
                        seen += 1
                        if j in likes:
                            hits += 1
                    if seen == n:
                        break
                recall.append(hits / float(len(likes)))
                bar.update()
            print numpy.mean(recall), ",", len(sampled_users) / total_time

        ## Brute force
    subsample_users = sampled_users[0:2000]
    bar = pyprind.ProgBar(len(subsample_users))
    total_time = 0.0
    recall = []
    for u in subsample_users:
        u_vector = U[u]
        variance = 2 * (Var[0, u] ** 2)
        likes = valid[u]
        to_exclude = train[u]
        clock = time.clock()
        scores = numpy.square(u_vector - V).sum(1) / variance - bias
        top_items = numpy.argpartition(scores, n + len(train))
        total_time += time.clock() - clock
        seen = 0
        hits = 0
        for j in top_items[numpy.argsort(scores[top_items])]:

            if j in to_exclude:
                continue
            else:
                seen += 1
                if j in likes:
                    hits += 1
            if seen == n:
                break
        recall.append(hits / float(len(likes)))
        bar.update()
        bar.update()
    print len(subsample_users) / total_time


def eval_mf(id=2617998461777000254, n=10, n_samples=-1):
    model, train, valid, test, exclude = load_model(id)
    # preprocess U, V
    model = model.model
    n_items = model.item_embeddings.shape[0]
    n_users = model.user_embeddings.shape[0]
    V = numpy.concatenate(
        (model.item_embeddings, model.item_biases.reshape((n_items, 1))), axis=1)
    U = numpy.concatenate(
        (model.user_embeddings, numpy.ones((n_users, 1), dtype="float32")), axis=1)
    U = U / (numpy.square(U).sum(1) ** 0.5).reshape((n_users, 1))
    M = numpy.square(V).sum(1).max() ** 0.5
    V = V * 0.83 /  M
    V_norm = (V ** 2).sum(1) ** 0.5
    half = numpy.zeros((n_users, 1), dtype="float32") + 0.5
    for m in range(3):
        V = numpy.concatenate((V, (V_norm ** ((m + 1) * 2)).reshape((n_items, 1))),axis=1)
        U = numpy.concatenate((U, half), axis=1)

    users = valid.keys()
    numpy.random.seed(1)
    if n_samples > 0:
        sampled_users = numpy.random.choice(users, n_samples)
    else:
        sampled_users = users
    for n_trees in [10, 50, 100, 200, 400]:
        t = AnnoyIndex(V.shape[1], metric="euclidean")  # Length of item vector that will be indexed
        # build trees
        for i in xrange(n_items):
            t.add_item(i, V[i])
        t.build(n_trees)  # 10 trees
        print "Tree built: " + str(n_trees)

        for n_ann_samples in [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            print "N_Ann_Samples" + " " + str(n_ann_samples)
            recall = []
            bar = pyprind.ProgBar(len(sampled_users))
            total_time = 0.0
            for u in sampled_users:
                u_vector = U[u]
                likes = valid[u]
                to_exclude = train[u]
                clock = time.clock()
                close_items, distances = t.get_nns_by_vector(u_vector, n + len(to_exclude),

                                                             search_k=n_ann_samples, include_distances=True)
                total_time += time.clock() - clock
                distances = numpy.square(distances)
                scores = distances
                hits = 0
                seen = 0
                all_scores = scores

                for index in numpy.argsort(all_scores):
                    j = close_items[index]
                    if j in to_exclude:
                        continue
                    else:
                        seen += 1
                        if j in likes:
                            hits += 1
                    if seen == n:
                        break
                recall.append(hits / float(len(likes)))
                bar.update()
            print numpy.mean(recall), ",", len(sampled_users) / total_time

    ## Brute force
    V = model.item_embeddings
    U = model.user_embeddings
    bias = model.item_biases

    subsample_users = sampled_users[0:2000]
    bar = pyprind.ProgBar(len(subsample_users))
    total_time = 0.0
    recall = []
    for u in subsample_users:
        u_vector = U[u]
        likes = valid[u]
        to_exclude = train[u]
        clock = time.clock()
        scores = numpy.square(u_vector * V).sum(1) + bias
        top_items = numpy.argpartition(scores, n + len(train))
        total_time += time.clock() - clock
        seen = 0
        hits = 0
        for j in top_items[numpy.argsort(scores[top_items])]:

            if j in to_exclude:
                continue
            else:
                seen += 1
                if j in likes:
                    hits += 1
            if seen == n:
                break
        recall.append(hits / float(len(likes)))
        bar.update()
        bar.update()
    print len(subsample_users) / total_time
