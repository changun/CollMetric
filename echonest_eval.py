from model import *
from utils import *
import theano.misc.pkl_utils


# create datasets
user_dict_l = echonest(5)
n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
    user_dict_l, [3, 1, 1])

def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=2500, validation_frequency=100, n_epochs=10000000, adagrad=True)
    save("EchoNest", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, count_thres=5)


# files = [ 'Echonest_KBPR_N_Factors.p']
# save_batch(files, "EchoNest", n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l, count_thres=5)

import gc
## LightFM
# for n_factors in ( 100,):
#     model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0, lambda_v=1E-5, loss="warp")
#     train_and_save(model)
#     del model
#     gc.collect()

for n_factors in (100,):
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=100, lambda_bias=10, max_norm=1.0, learning_rate=0.1, warp_count=20,  lambda_cov=1)
    train_and_save(model)
    del model
    gc.collect()


for n_factors in (100,):
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=100, lambda_bias=10, max_norm=1.0, learning_rate=0.1, warp_count=20,  lambda_cov=100)
    train_and_save(model)
    del model
    gc.collect()

for n_factors in (100,):
    model = KBPRModel(n_factors, n_users_l, n_items_l,
                      margin=0.5, lambda_variance=1000, lambda_bias=10, max_norm=1.0, learning_rate=0.1, warp_count=20,  lambda_cov=10)
    train_and_save(model)
    del model
    gc.collect()