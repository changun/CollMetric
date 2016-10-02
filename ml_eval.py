from model import *
from utils import *
import theano.misc.pkl_utils

# create datasets
user_dict_l, features_l, labels, to_title = movielens20M(min_rating=0.0, user_rating_count=10, tag_freq_thres=20, use_director=False)
n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
    user_dict_l, [3, 1, 1])

files = ["MovieLens0_LightFM_N_Factors.p",
         "MovieLens0_KBPR_N_Factors.p", ]

save_batch(files, "MovieLens", n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l,
           subject_thres=5, min_rating=0.0, user_rating_count=10, tag_freq_thres=20, use_director=False)

files4 = [
         "MovieLens4_KBPR_N_Factors.p",
         "MovieLens4_VKBPR_N_Factors.p"]

save_batch(files4, "MovieLens", n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l,
           subject_thres=5, min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)


def train_fn(model):
    return early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True)[1]


def save(method, model, name):
    import os, shutil
    results = []
    try:
        results = theano.misc.pkl_utils.load(open(name, "rb"))
    except Exception:
        pass

    results.append({"method": method, "model": model})
    tmp = os.tempnam()
    theano.misc.pkl_utils.dump(results, open(tmp, "wb"))
    shutil.move(tmp, name)
    print str(theano.misc.pkl_utils.load(open(name, "rb"))) + "\n"


## LightFM
name = "MovieLens0_LightFM_N_Factors.p"
for n_factors in (10, 40, 70, 100):
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
    model = train_fn(model)
    save("lightfm", model, name)

# ## LightFM Features
# name = "MovieLens4_LightFM_Feature_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, features_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)

## KBPR
name = "MovieLens0_KBPR_N_Factors.p"
for n_factors in (10, 40, 70, 100):
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=10.0, lambda_bias=0.1, max_norm=1.0, warp_count=20, lambda_cov=10)
    model = train_fn(model)
    save("kbpr", model, name)

#
# for n_factors in (10, 40, 70, 100):
#     model = KBPRModel(n_factors, n_users_l, n_items_l,
#                       margin=0.5, lambda_variance=10.0, lambda_bias=0.1, max_norm=1.0, warp_count=20, lambda_cov=10)
#     model = train_fn(model)
#     save("kbpr", model, name)
#
#
# name = "MovieLens4_KBPR_N_Factors.p"
# for n_factors in (70, 100):
#     model = KBPRModel(n_factors, n_users_l, n_items_l,
#                       margin=0.5, lambda_variance=10.0, lambda_bias=0.10, max_norm=1.0, warp_count=20)
#     model = train_fn(model)
#     save("kbpr", model, name)
# #
name = "MovieLens4_VKBPR_N_Factors.p"
for n_factors in (10, 40, 70):
    import scipy.sparse
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l.toarray(),
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=1.0, n_layers=2, width=64,
                             lambda_v_off=0.1, lambda_variance=10, lambda_bias=1,
                             embedding_rescale=0.5, warp_count=20)
    model = train_fn(model)
    save("vkbpr_mlp", model, name)
