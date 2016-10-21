from model import *
from utils import *
import theano.misc.pkl_utils

# create datasets
user_dict_l, features_l, labels, to_title = citeulike(5)
n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
    user_dict_l, [3, 1, 1])

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True)
    save("Citeulike", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, tag_thres=5)


m = KBPRModel(100, n_users_l, n_items_l,
                         lambda_u=0, lambda_v=0, use_bias=True,  max_norm=1.0, lambda_cov=100,

                         margin=.5, learning_rate=.1,
                          batch_size=3000, warp_count=20)
train_and_save(m)
# ## LightFM
# name = "Citeulike_LightFM_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
    # model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
    # train_and_save(model)
    # model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-6, lambda_v=1E-6, loss="warp")
    # train_and_save(model)
    # model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=1E-5, loss="warp")
    # train_and_save(model)


model = BPRModel(100, n_users_l, n_items_l, margin=1.0, max_norm=1.0,
                  learning_rate=0.1, loss_f="hinge",
                 warp_count=20, batch_size=10000, )

train_and_save(model)
model = KBPRModel(100, n_users_l, n_items_l,
                  margin=1.0, lambda_variance=100, lambda_bias=10, max_norm=1.0, learning_rate=0.1, warp_count=20, batch_size=10000, lambda_cov=100)
model.loss_f = "sigmoid"
train_and_save(model)
#
# # LightFM Features
# name = "Citeulike_LightFM_Feature_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
#     import scipy.sparse
#     model = LightFMModel(n_factors, n_users_l, n_items_l, scipy.sparse.csc_matrix(features_l, dtype="float32")[0:n_items_l,:], lambda_u=1E-6, lambda_v=1E-6, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)
#
# # # KBPR
# name = "Citeulike_KBPR_N_Factors.p"
# for n_factors in (10,):
#     model = KBPRModel(n_factors, n_users_l, n_items_l, margin=1.0, lambda_variance=100, lambda_bias=10, max_norm=1.0, learning_rate=0.1, warp_count=20, batch_size=10000, lambda_cov=100)
#     model = train_fn(model)
#     save("kbpr", model, name)
#
# name = "Citeulike_VKBPR_N_Factors.p"
# for n_factors in (10, 40, 70, 100):
#     import scipy.sparse
#     model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, scipy.sparse.csc_matrix(features_l, dtype="float32")[0:n_items_l,:],
#                              lambda_weight_l1=0.0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
#                              lambda_v_off=1, lambda_variance=100, lambda_bias=10,
#                              embedding_rescale=1, warp_count=20, learning_rate=0.1, lambda_cov=10, batch_size=10000)
#     model = train_fn(model)
#     save("kbpr", model, name)

name = "Citeulike_VKBPR_N_Factors_K.p"
for n_factors, model_id in ([100, 7058420260225614648], [70,2699180007298258092], [40,	-8538943969636385630], [10, 2806190959102487375]):
    for K in (2,):
        model = load_model(model_id)[0]
        #new_U = model.kmean(train_dict_l, K)
        new_model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, scipy.sparse.csc_matrix(features_l, dtype="float32").toarray(),

                                     V_mlp=model.V_mlp, learning_rate=0.01, margin=1.0, K=K,
                             lambda_weight_l1=0, lambda_weight_l2=0.0,
                             lambda_v_off=1,lambda_bias=10, lambda_variance=100,  lambda_density=1,
                             embedding_rescale=1, warp_count=20, batch_size=10000,  lambda_cov=10,)
        train_and_save(new_model)
        import gc
        gc.collect()

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=5, validation_frequency=1, n_epochs=10000000, adagrad=True)
    save("Citeulike", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, tag_thres=5)

train_and_save(model)
for n_factors in (100,):
    #model = WRMF(n_users_l, n_items_l, n_factors, 1, 0.01, 0, 0)
    #train_and_save(model)
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

