from utils import *
from model import *
from sklearn.preprocessing import *
n_users, n_items, _train, _valid, _test, U_features, U_features_ut, U_features_ut_idf = yelp()
thres = 3
user_dict = merge_dict(merge_dict(_train, _valid), _test)
numpy.random.seed(1)
target_users = [user for user, items in user_dict.items() if len(items) > thres]
test_users = set(numpy.random.choice(target_users, 3000, replace=False))
valid_users = set(numpy.random.choice(list(set(target_users) - test_users), 1000, replace=False))
train_users = set([u for u in user_dict.keys() if u not in test_users and u not in valid_users])
def gen_dict(user_dict, test_users, keep_n, train_n, exclude_users = None):
    train_dict = {}
    test_dict = {}
    for u, items in user_dict.items():
        if u in test_users:
            numpy.random.seed(u)
            keep = numpy.random.choice(list(items), replace=False, size=keep_n)
            if train_n > 0:
                train_dict[u] = numpy.random.choice(keep, replace=False, size=train_n)
            test_dict[u] = items - set(keep)
        elif exclude_users is not None and u in exclude_users:
            continue
        else:
            train_dict[u] = items

    return train_dict, test_dict

# parameter selection

results = []
for train_n in [1]:
    train, valid = gen_dict(user_dict, valid_users, keep_n=3, train_n=train_n, exclude_users=test_users)
    train_valid, test = gen_dict(user_dict, test_users, keep_n=3, train_n=train_n)
    for model_fn in [#lambda K, active_users: UserKNN(n_users, n_items, K=K, active_users=list(active_users)),
                    lambda K, active_users: ProfileKNN(n_users, n_items, U_features_ut, K=K, active_users=list(active_users))
                     ]:
        # parameter selection
        best_recall = 0
        best_K = 0
        for K in [100, 200, 300, 400, 500, 600, 700]:#
            model = model_fn(K, train_users)
            print model
            model.train(train)
            recall = model.recall(valid,train, ns=[10])[0][0]
            if recall > best_recall:
                best_K = K
                best_recall = recall
        best_model = model_fn(best_K, valid_users | train_users)
        print "=======================================\nBest Model" + str(best_model)
        best_model.train(train_valid)
        results.append({"recalls": best_model.recall(test, train, ns=[50, 20, 10, 5, 1]),
                        "train_n": train_n,
                        "best_K": best_K,
                        "model": best_model})

p_model = Popularity(n_users, n_items)
train_valid, test = gen_dict(user_dict, test_users, keep_n=3, train_n=0)
p_model.train(train_valid)
results.append({"recalls": p_model.recall(test, train_valid, ns=[50, 20, 10, 5, 1]),
                "model": p_model})





