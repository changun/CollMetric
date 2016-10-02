from model import *
from utils import *
import theano.misc.pkl_utils

# create datasets
user_dict_l, features_l, labels, to_title = lastfm(user_thres=10)
n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
    user_dict_l, [3, 1, 1])


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

files = ["LastFM_LightFM_N_Factors_10.p",  "LastFM_LightFM_Feature_N_Factors_10.p",]

save_batch(files, "LastFM", n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l,user_thres=10)


### LightFM
# name = "LastFM_LightFM_N_Factors_10.p"
# for n_factors in (10,40,70, 100,):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)
# for n_factors in (10,40,70, 100,):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=1E-5, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)
## LightFM Features
# name = "LastFM_LightFM_Feature_N_Factors_10.p"
# for n_factors in (10, 40, 70, 100,):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, features_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)
#

# ## KBPR
# name = "LastFM_KBPR_N_Factors_10.p"
# for n_factors in (40, 70, 100):
#     model = KBPRModel(n_factors, n_users_l, n_items_l, margin=1.0, lambda_variance=10, lambda_bias=1, max_norm=1.0, warp_count=20)
#     model = train_fn(model)
#     save("kbpr", model, name)
#
# #
# # for n_factors in (40, 70, 100):
# #     model = KBPRModel(n_factors, n_users_l, n_items_l, margin=0.5, lambda_variance=1, lambda_bias=0.1, max_norm=1.0, warp_count=20)
# #     model = train_fn(model)
# #     save("kbpr", model, name)
# #
# #
# for n_factors in (10, 40, 70, 100):
#     model = KBPRModel(n_factors, n_users_l, n_items_l, margin=0.5, lambda_variance=1, lambda_bias=1, max_norm=1.0, warp_count=20, lambda_cov=10, learning_rate=0.01)
#     model = train_fn(model)
#     save("kbpr", model, name)
#
name = "LastFM_VKBPR_N_Factors_10.p"
for n_factors in (10, 40, 70, 100):
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l.toarray(),
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, lambda_variance=1, lambda_bias=1, margin=1.0,
                             embedding_rescale=0.1)

    model = train_fn(model)
    save("vkbpr", model, name)

import scipy.sparse
features_l = features_l.toarray()
name = "LastFM_VKBPR_N_Factors_K_10.p"
for n_factors in (40, 70, 100,):
    for K in (2,):
        model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5, lambda_bias=1, lambda_variance=1, warp_count=20,
                             embedding_rescale=0.1)
        model = train_fn(model)
        new_U = model.kmean(train_dict_l, K)
        new_model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l, U=new_U, V=V,
                                     V_mlp=V_mlp, learning_rate=0.01, margin=0.9, K=K,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, lambda_bias=1, lambda_variance=0.1, lambda_density=1, warp_count=10,
                             embedding_rescale=0.1)
        new_model = train_fn(new_model)
        save("vkbpr", new_model, name)