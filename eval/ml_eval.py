from model import *
from utils import *
import theano.misc.pkl_utils


user_dict_l, features_l, labels, to_title = movielens20M(min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)
n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
     user_dict_l, [3, 1, 1])


def load_data():
    from pymongo import MongoClient
    import gridfs, cPickle
    db = MongoClient().features
    fs = gridfs.GridFS(db)
    model, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l = load_model(-1438456719658488636)
    n_users_l = model.n_users
    n_items_l = model.n_items
    cold_dict_l = None
    popular_l = None
    cold_l = None
    features_f = cPickle.load(fs.get(db.fs.files.find_one({"dataset": "MovieLens"})["_id"]))
    return model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
           popular_l,  cold_l, features_f

model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
popular_l,  cold_l, features_f  = load_data()


def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True, dynamic=True)
    save("MovieLens", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, subject_thres=5, min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)

## LightFM
for n_factors in (100, 10, 40, 70):
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=1E-5, loss="warp")
    train_and_save(model)
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-4, lambda_v=1E-4, loss="warp")
    train_and_save(model)

## KBPR
for n_factors in (100,):
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=10.0, lambda_bias=0.1, max_norm=1.0, warp_count=20, lambda_cov=10)
    train_and_save(model)
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=10.0, lambda_bias=1, max_norm=1.0, warp_count=20, lambda_cov=100)
    train_and_save(model)
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=100.0, lambda_bias=0.1, max_norm=1.0, warp_count=20, lambda_cov=10)
    train_and_save(model)

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
