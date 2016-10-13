from model import *
from utils import *
import theano.misc.pkl_utils

#create datasets
# user_dict_l, features_l, topics, labels, to_title = medium()
# n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
#      user_dict_l, [3, 1, 1])
# features_l = topics
# from pymongo import MongoClient
# import gridfs
#
# db = MongoClient().features
# fs = gridfs.GridFS(db)
# import io, cPickle
# with io.BytesIO() as byte:
#      theano.misc.pkl_utils.dump(features_l, byte)
#      fs.put(byte.getvalue(), _id=hash(byte.getvalue()), dataset="Medium", with_topics=True)
#      print hash(byte.getvalue())
#
def load_data():
    from pymongo import MongoClient
    import gridfs, cPickle
    db = MongoClient().features
    fs = gridfs.GridFS(db)
    model, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l = load_model(-6847242372680792284)
    n_users_l = model.n_users
    n_items_l = model.n_items
    cold_dict_l = {}
    popular_l = {}
    cold_l = {}
    features_f = theano.misc.pkl_utils.load(fs.get(2426373644027780065))
    return model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
           popular_l,  cold_l, features_f

model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
popular_l,  cold_l, features_l  = load_data()

# import scipy.sparse
# features_l = scipy.sparse.csc_matrix(features_l, dtype="float32")

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True, )
    save("Medium", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, with_topics=True)


for n_factors in (100, 10, 40, 70):
    n_factors = n_factors / 2
    model = VisualBPR(n_factors, n_users_l, n_items_l, features_l,
                            lambda_u=0.1, lambda_v=0.1, lambda_bias=1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                            margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)

    train_and_save(model)
    model = VisualBPR(n_factors, n_users_l, n_items_l, features_l,
                            lambda_u=0.1, lambda_v=1.0, lambda_bias=0.1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                             margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)

    train_and_save(model)
    model = VisualBPR(n_factors, n_users_l, n_items_l, features_l,
                            lambda_u=1, lambda_v=1, lambda_bias=0.1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                             margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)

    train_and_save(model)




for n_factors in (100, 10, 40, 70):
    model = VisualOffsetBPR(n_factors, n_users_l, n_items_l, features_l,
                            lambda_u=0.01, lambda_v=0.01, lambda_bias=0.1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                            lambda_v_off=1, margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)
    train_and_save(model)
    model = VisualOffsetBPR(n_factors, n_users_l, n_items_l, features_l,
                            lambda_u=1, lambda_v=1, lambda_bias=0.1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=2, width=256,
                            lambda_v_off=1, margin=1.0, learning_rate=.05,
                            embedding_rescale=0.1, batch_size=200000)
    train_and_save(model)

# name = "Medium_VKBPR_N_Factors.p"
model = VisualFactorKBPR(100, n_users_l, n_items_l, features_l,
                         lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=1, n_layers=3, width=256,
                         lambda_v_off=1.0, lambda_bias=10, lambda_variance=10.0, lambda_cov=100,
                         embedding_rescale=0.1, warp_count=20, batch_size=200000, learning_rate=0.1, margin=0.5)
train_and_save(model)

model = VisualFactorKBPR(100, n_users_l, n_items_l, features_l,
                         lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=1, n_layers=3, width=256,
                         lambda_v_off=1.0, lambda_bias=10, lambda_variance=5.0, lambda_cov=100,
                         embedding_rescale=0.1, warp_count=20, batch_size=200000, learning_rate=0.1, margin=0.5)
train_and_save(model)

model = VisualFactorKBPR(100, n_users_l, n_items_l, features_l,
                         lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=1, n_layers=3, width=256,
                         lambda_v_off=1.0, lambda_bias=10, lambda_variance=10.0, lambda_cov=10,
                         embedding_rescale=0.1, warp_count=20, batch_size=200000, learning_rate=0.1, margin=0.5)
train_and_save(model)
for n_factors in (100, 10, 40, 70):
    features_l = features_l
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1.0, lambda_bias=10, lambda_variance=100.0, lambda_cov=10,
                             embedding_rescale=0.1, warp_count=20, batch_size=200000, learning_rate=0.05, margin=0.5)
    train_and_save(model)
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1.5, lambda_bias=10, margin=0.5,
                             embedding_rescale=0.04, warp_count=20, batch_size=200000, learning_rate=0.05, lambda_cov=10)
    train_and_save(model)
# for n in [100, 70, 40, 10]:
#     model = KBPRModel(n, n_users_l, n_items_l, margin=0.5, lambda_variance=100,
#                       lambda_bias=0.1, max_norm=1.0, warp_count=20, lambda_cov=100, batch_size=50000)
#     train_and_save(model)

# ## LightFM
# name = "Medium_LightFM_N_Factors.p"
# for n_factors in (100, 70, 40, 10):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0.0, lambda_v=1E-5, loss="warp")
#     train_and_save(model)

## LightFM Features
name = "Medium_LightFM_Feature_N_Factors.p"
for n_factors in (70, 100,):
    model = LightFMModel(n_factors, n_users_l, n_items_l, features_l, lambda_u=0.0, lambda_v=1E-5, loss="warp")
    model = train_and_save(model)

#
# ## KBPR
# name = "Medium_KBPR_N_Factors.p"
# for n_factors in (10, 40, 70, 100):
#     model = KBPRModel(n_factors, n_users_l, n_items_l, margin=0.5, lambda_variance=10, lambda_bias=10, max_norm=1.0, warp_count=20)
#     model = train_fn(model)
#     save("kbpr", model, name)

#
name = "Medium_KBPR_N_Factors_K.p"
# for n_factors in (100,):
#     for K in (2,):
#         model = KBPRModel(n_factors/K, n_users_l, n_items_l, margin=0.5, lambda_bias=10, lambda_variance=10)
#         model = train_fn(model)
#         new_U = model.kmean(train_dict_l, K)
#         new_model = KBPRModel(n_factors / K, n_users_l, n_items_l, U=new_U, V=model.V.get_value(),
#                               lambda_bias=10, lambda_variance=0.1, K=K, learning_rate=0.01, margin=0.7, lambda_density=1)
#         new_model = train_fn(new_model)
#         save("kbpr", new_model, name)
# #

#     model = train_fn(model)
#     save("vkbpr_mlp", model, name)


name = "Medium_VKBPR_N_Factors_K.p"
for n_factors in ( 100,):
    for K in (2,):
        model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l.toarray(),
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5, lambda_bias=10, lambda_variance=10,
                             embedding_rescale=0.04, warp_count=20, batch_size=100000)
        model = train_fn(model)
        new_U = model.kmean(train_dict_l, K)
        new_model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_l.toarray(), U=new_U, V=model.V.get_value(),
                                     V_mlp=model.V_mlp, learning_rate=0.01,margin=0.7, K=K,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1,lambda_bias=10, lambda_variance=1,  lambda_density=1,
                             embedding_rescale=0.04, warp_count=20, batch_size=80000)
        new_model = train_fn(new_model)
        save("vkbpr", new_model, name)

