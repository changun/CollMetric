from model import *
from utils import *
import theano.misc.pkl_utils

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

for n_factors in (100, 10, 40, 70):
    model = VisualOffsetBPR(n_factors, n_users_l, n_items_l, features_f,
                            lambda_u=0.1, lambda_v=0.1, lambda_bias=0.1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                            lambda_v_off=1, margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)
    train_and_save(model)


### LightFM
# name = "Flickr_LightFM_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
#     model = LightFMModel(n_factors, n_users_f, n_items_f, lambda_u=0.0, lambda_v=0.0, loss="warp", learning_rate=0.01)
#     model = train_fn(model)
#     save("lightfm", model, name)
#
# name = "Flickr_LightFM_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
#     model = LightFMModel(n_factors, n_users_f, n_items_f, lambda_u=0, lambda_v=0, loss="bpr")
#     model = train_fn(model)
#     save("lightfm", model, name)
#
#
#
# name = "Flickr_Hinge_N_Factors.p"
# for n_factors in (10, 40, 70,  100):
#     model = BPRModel(n_factors, n_users_f, n_items_f, margin=0.5, loss_f="hinge",
#                      lambda_b=1.0, lambda_u=1.0, lambda_v=1.0, warp_count=10)
#     model = train_fn(model)
#     save("hinge", model, name)
# for n_factors in (10, 40, 70,  100):
#     model = BPRModel(n_factors, n_users_f, n_items_f, margin=0.5, loss_f="hinge",
#                      lambda_b=1.0, lambda_u=10.0, lambda_v=10.0, warp_count=10)
#     model = train_fn(model)
#     save("hinge", model, name)
# for n_factors in (10, 40, 70,  100):
#     model = BPRModel(n_factors, n_users_f, n_items_f, margin=0.5, loss_f="hinge",
#                      lambda_b=10.0, lambda_u=10.0, lambda_v=10.0, warp_count=10)
#     model = train_fn(model)
#     save("hinge", model, name)
#
# for n_factors in (10, 40, 70,  100):
#     model = BPRModel(n_factors, n_users_f, n_items_f, margin=0.5, loss_f="hinge",
#                      lambda_b=100.0, lambda_u=100.0, lambda_v=100.0, warp_count=10)
#     model = train_fn(model)
#     save("hinge", model, name)
#
#

# ## KBPR
# name = "Flickr_KBPR_N_Factors.p"
# for n_factors in (10, 40, 70,  100):
#     model = KBPRModel(n_factors, n_users_f, n_items_f, margin=0.5, negative_sample_choice="max", lambda_cov=100)
#     model = train_fn(model)
#     save("kbpr", model, name)
# #
# # name = "Flickr_KBPR_N_Factors_K.p"
# # for n_factors in (10, 40, 70, 100):
# #     for K in (2,):
# #         model = KBPRModel(n_factors/K, n_users_f, n_items_f, margin=0.5, lambda_bias=10, lambda_variance=100)
# #         model = train_fn(model)
# #         new_U = model.kmean(train_dict_f, K)
# #         new_model = KBPRModel(n_factors / K, n_users_f, n_items_f, U=new_U, V=model.V.get_value(),
# #                               lambda_bias=10, lambda_variance=100, K=K, learning_rate=0.01, margin=1.0)
# #         train_fn(new_model)
# #         save("kbpr", new_model, name)
#
# #
name = "Flickr_VKBPR_N_Factors.p"
for n_factors in (100,):
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5,
                             embedding_rescale=0.1, lambda_cov=100, warp_count=20, learning_rate=0.05, lambda_bias=0.1)
    train_and_save(model)
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=1.0,
                             embedding_rescale=0.1, lambda_cov=100, warp_count=20, learning_rate=0.05, lambda_bias=0.1)
    train_and_save(model)
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5,
                             embedding_rescale=0.1, lambda_cov=100, warp_count=20, learning_rate=0.05, lambda_bias=10)
    train_and_save(model)


#
#
# for n_factors in (10, 40, 70, 100):
#     model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f,
#                              lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
#                              lambda_v_off=1, margin=0.5,
#                              embedding_rescale=0.1, lambda_cov=1)
#
#     model = train_fn(model)
#     save("vkbpr", model, name)




name = "Flickr_VKBPR_N_Factors_K.p"
for n_factors, model_id in ([100, -7688251035354585734],):
    model = load_model(model_id)[0]
    V_mlp = model.V_mlp

    new_U = model.kmean(train_dict_l, 2)
    new_model = VisualFactorKBPR(n_factors, n_users_l, n_items_l,
                                 features_f,
                                 V=model.V.get_value(), U=new_U,
                                 V_mlp=V_mlp, learning_rate=0.01, margin=1.0, K=2,
                                 lambda_weight_l1=0, lambda_weight_l2=0.0,
                                 lambda_v_off=1, lambda_bias=10, lambda_variance=100, lambda_density=1,
                                 embedding_rescale=0.1, warp_count=20, batch_size=10000, lambda_cov=10, )
    train_and_save(new_model)
    import gc
    gc.collect()
    new_model = VisualFactorKBPR(n_factors, n_users_l, n_items_l,
                                 features_f,
                                 V=model.V.get_value(), U=new_U,
                                 V_mlp=V_mlp, learning_rate=0.01, margin=1.0, K=2,
                                 lambda_weight_l1=0, lambda_weight_l2=0.0,
                                 lambda_v_off=1, lambda_bias=10, lambda_variance=1, lambda_density=1,
                                 embedding_rescale=0.1, warp_count=20, batch_size=10000, lambda_cov=10, )
    train_and_save(new_model)
