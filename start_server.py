from model import *
from utils import *
from server import *
import theano.misc.pkl_utils

# create datasets

server = SimpleHttpServer("0.0.0.0", 8080)

user_dict_f, features_f, id_photos = flickr(features=True)
n_items_f, n_users_f, train_dict_f, valid_dict_f, test_dict_f, exclude_dict_f, cold_dict_f, popular_f, cold_f = preprocess(
    user_dict_f, [3, 1, 1])

server.addModel("Flickr", theano.misc.pkl_utils.load("Flickr_50_2_lambda_bias_10_var_100.p"), flickr_details(id_photos), train_dict_f, test_dict_f)


user_dict_m, features_m, labels_m, to_title_m = medium()
n_items_m, n_users_m, train_dict_m, valid_dict_m, test_dict_m, exclude_dict_m, cold_dict_m, popular_m, cold_m = preprocess(
    user_dict_m, [3, 1, 1])

server.addModel("medium", theano.misc.pkl_utils.load( "Medium_50_F2_Pop_v_Cold.p"), medium_details(to_title_m), train_dict_m, test_dict_m)
