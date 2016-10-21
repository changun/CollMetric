from model import *
from utils import *
import gzip
# # create datasets
# user_dict_f, features_f, id_photos = flickr(features=True)
# n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
#     user_dict_f, [3, 1, 1])

def load_data():
    from pymongo import MongoClient
    import gridfs
    db = MongoClient().features
    fs = gridfs.GridFS(db)
    model, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l = load_model(-1506314373122977297)
    n_users_l = model.n_users
    n_items_l = model.n_items
    cold_dict_l = None
    popular_l = None
    cold_l = None
    features_f = numpy.load(fs.get(db.fs.files.find_one({"dataset": "Flickr"})["_id"]))
    return model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
           popular_l,  cold_l, features_f

model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
popular_l,  cold_l, features_f  = load_data()

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True)
    save("Flickr", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l)


m = KBPRModel(100, n_users_l, n_items_l,
                         lambda_u=0, lambda_v=0, use_bias=False,  max_norm=1.0, lambda_cov=100,

                         margin=.5, learning_rate=.1,
                          batch_size=100000, warp_count=20)

for n_factors in (100, ):
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                      lambda_u=0, lambda_v=0, lambda_bias=10, lambda_variance=100, max_norm=1.0, lambda_cov=100,
                      lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                      margin=.5, learning_rate=.05,
                      embedding_rescale=0.1, batch_size=400000, warp_count=20)

    train_and_save(model)
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                             lambda_u=0, lambda_v=0, lambda_bias=10, lambda_variance=100, max_norm=1.0, lambda_cov=100,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=4, width=256,
                             margin=.5, learning_rate=.05,
                             embedding_rescale=0.1, batch_size=400000, warp_count=20)

    train_and_save(model)

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=5, validation_frequency=1, n_epochs=10000000, adagrad=True)
    save("Flickr", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l)


for n_factors in (100,):
    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.01, 0, 0)
    train_and_save(model)
    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.01, 0.01, 0.01)
    train_and_save(model)
    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.01, 0.01, 0.01)
    train_and_save(model)

    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.1, 0, 0)
    train_and_save(model)
    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.1, 0.01, 0.01)
    train_and_save(model)
    model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.1, 0.01, 0.01)
    train_and_save(model)

