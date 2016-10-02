from model import *
from utils import *
import theano.misc.pkl_utils

# create datasets
user_dict_f, features_f, id_photos = flickr(features=True)
n_items_f, n_users_f, train_dict_f, valid_dict_f, test_dict_f, exclude_dict_f, cold_dict_f, popular_f, cold_f = preprocess(
    user_dict_f, [3, 1, 1])

files = ["Flickr_KBPR_N_Factors.p",  "Flickr_VKBPR_N_Factors.p",
"Flickr_VKBPR_N_Factors_K.p", "Flickr_KBPR_N_Factors_K.p",
         ]

save_batch(files, "Flickr", n_users_f, n_items_f, train_dict_f, valid_dict_f, test_dict_f, exclude_dict_f, cold_dict_f, popular_f, cold_f,)


def train_fn(model):
    return early_stop(model, train_dict_f, lambda m: -m.recall(valid_dict_f, train_dict_f, n_users=3000)[0][0],
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
name = "Flickr_KBPR_N_Factors.p"
for n_factors in (10, 40, 70,  100):
    model = KBPRModel(n_factors, n_users_f, n_items_f, margin=0.5, negative_sample_choice="max", lambda_cov=100)
    model = train_fn(model)
    save("kbpr", model, name)
#
# name = "Flickr_KBPR_N_Factors_K.p"
# for n_factors in (10, 40, 70, 100):
#     for K in (2,):
#         model = KBPRModel(n_factors/K, n_users_f, n_items_f, margin=0.5, lambda_bias=10, lambda_variance=100)
#         model = train_fn(model)
#         new_U = model.kmean(train_dict_f, K)
#         new_model = KBPRModel(n_factors / K, n_users_f, n_items_f, U=new_U, V=model.V.get_value(),
#                               lambda_bias=10, lambda_variance=100, K=K, learning_rate=0.01, margin=1.0)
#         train_fn(new_model)
#         save("kbpr", new_model, name)

#
name = "Flickr_VKBPR_N_Factors.p"
for n_factors in (10, 40, 70, 100):
    model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5,
                             embedding_rescale=0.1, lambda_cov=10)

    model = train_fn(model)
    save("vkbpr", model, name)


for n_factors in (10, 40, 70, 100):
    model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5,
                             embedding_rescale=0.1, lambda_cov=1)

    model = train_fn(model)
    save("vkbpr", model, name)




name = "Flickr_VKBPR_N_Factors_K.p"
for n_factors in (40, 70, 100,):
    for K in (2,):
        model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5, lambda_bias=10, lambda_variance=100,
                             embedding_rescale=0.1)
        model = train_fn(model)
        new_U = model.kmean(train_dict_f, K)
        new_model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f, U=new_U, V=model.V.get_value(),
                                     V_mlp=model.V_mlp, learning_rate=0.01, margin=1.3, K=K,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1,lambda_bias=10, lambda_variance=0.1, lambda_density=1,
                             embedding_rescale=0.1)
        new_model = train_fn(new_model)
        save("vkbpr", new_model, name)

