from model import *
from utils import *
import theano.misc.pkl_utils

# create datasets
user_dict_f, features_f, _ = flickr(features=True)
n_items_f, n_users_f, train_dict_f, valid_dict_f, test_dict_f, exclude_dict_f, cold_dict_f, popular_f, cold_f = preprocess(
    user_dict_f, [3, 1, 1])


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


# ### LightFM
# name = "Flickr_LightFM_N_Factors.p"
# for n_factors in (10, 40, 70, 100,):
#     model = LightFMModel(n_factors, n_users_f, n_items_f, lambda_u=0.0, lambda_v=0.0, loss="warp")
#     model = train_fn(model)
#     save("lightfm", model, name)
# ## KBPR
name = "Flickr_KBPR_N_Factors.p"
for n_factors in (10, 40, 70,  100):
    model = KBPRModel(n_factors, n_users_f, n_items_f, margin=0.5)
    model = train_fn(model)
    save("kbpr", model, name)

name = "Flickr_VKBPR_N_Factors.p"
for n_factors in (10, 40, 70, 100):
    model = VisualFactorKBPR(n_factors, n_users_f, n_items_f, features_f,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=.5, n_layers=2, width=256,
                             lambda_v_off=1, margin=0.5,
                             embedding_rescale=0.1)

    model = train_fn(model)
    save("vkbpr", model, name)
