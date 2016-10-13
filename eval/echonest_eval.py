from model import *
from utils import *
import theano.misc.pkl_utils


# create datasets
#user_dict_l = echonest(5)
# n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
#     user_dict_l, [3, 1, 1])

def load_data():
    from pymongo import MongoClient
    import gridfs, cPickle
    db = MongoClient().features
    fs = gridfs.GridFS(db)
    model, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l = load_model(3871434674368471357)
    n_users_l = model.n_users
    n_items_l = model.n_items
    cold_dict_l = None
    popular_l = None
    cold_l = None
    #features_f = cPickle.load(fs.get(db.fs.files.find_one({"dataset": "MovieLens"})["_id"]))
    return model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
           popular_l,  cold_l, None

model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
popular_l,  cold_l, features_l  = load_data()


def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True)
    save("EchoNest", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, count_thres=5)


# files = [ 'Echonest_KBPR_N_Factors.p']
# save_batch(files, "EchoNest", n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l, count_thres=5)

import gc
## LightFM
# for n_factors in ( 100, 70, 40, 10):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-4, lambda_v=1E-4, loss="warp")
#     train_and_save(model)
#     del model
#     gc.collect()
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=1E-4, loss="warp")
#     train_and_save(model)
#     del model
#     gc.collect()
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=0, loss="warp")
#     train_and_save(model)
#     del model
#     gc.collect()
# train_and_save(model)
# for n_factors in (40, 10):
#     model = KBPRModel(n_factors, n_users_l, n_items_l,
#                       margin=0.5, lambda_variance=100, lambda_bias=10, max_norm=1.0, learning_rate=0.05,
#                       warp_count=20, lambda_cov=10, batch_size=400000)
#     train_and_save(model)
#     del model
#     gc.collect()


for n_factors in ( 40, 70):
    model = BPRModel(n_factors, n_users_l, n_items_l,
                            lambda_u=0.01, lambda_v=0.01, lambda_b=0.1,
                             margin=1.0, learning_rate=.01,
                             batch_size=200000, loss_f="sigmoid", warp_count=1)
    train_and_save(model)
    model = BPRModel(n_factors, n_users_l, n_items_l,
                     lambda_u=0.01, lambda_v=0.01, lambda_b=1,
                     margin=1.0, learning_rate=.01,
                     batch_size=200000, loss_f="sigmoid", warp_count=1)
    train_and_save(model)