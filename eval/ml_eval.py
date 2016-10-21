from model import *
from utils import *
import theano.misc.pkl_utils

#
# user_dict_l, features_l, labels, to_title = movielens20M(min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)
# from pymongo import MongoClient
# import gridfs
#
# db = MongoClient().features
# fs = gridfs.GridFS(db)
# import io, cPickle
# with io.BytesIO() as byte:
#      theano.misc.pkl_utils.dump(features_l, byte)
#      fs.put(byte.getvalue(), _id=hash(byte.getvalue()), dataset="MovieLens", with_topics=True)
#      print hash(byte.getvalue())

# n_items_l, n_users_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, popular_l, cold_l = preprocess(
#      user_dict_l, [3, 1, 1])

# #
# infos = theano.misc.pkl_utils.load(open("/data/dataset/MovieLens20MInfo.p", "rb"))
# import csv, re
# ls = csv.reader(open("/data/dataset/ml-20m/movies.csv"), delimiter=',', quotechar='"')
# ls.next()
# links = csv.reader(open("/data/dataset/ml-20m/links.csv"), delimiter=',', quotechar='"')
# links.next()
# import imdb
# ia = imdb.IMDb()
# for id, name, genres in ls:
#     imdb = links.next()[1]
#     year_re = re.search('\((\d+)\)', name)
#     if year_re is not None:
#         year = int(year_re.group(1))
#     else:
#         year = 1984
#     try:
#         if int(id) in infos and "imdbid" in infos[int(id)]["data"] and infos[int(id)]["data"]["imdbid"] == imdb:
#             continue
#         if int(id) in infos and infos[int(id)]["data"]["title"].upper() in name.upper() and int(infos[int(id)]["data"]["year"]) == year:
#             continue
#     except Exception as e:
#         print e
#
#     print "Correct " + id +" " + name + " " + imdb
#     m = ia.get_movie(imdb)
#     info = ia.get_movie_main(m.getID())
#     keywords = ia.get_movie_keywords(m.getID())
#     if 'data' in keywords and "keywords" in keywords["data"]:
#         info["data"]["keywords"] = keywords["data"]["keywords"]
#     info["data"]["imdbid"] = imdb
#     infos[int(id)] = info
#     print m
#
#
# f = open("/data/dataset/ml-20m/links.csv")
# f.next()
# info_set = {}
# import imdb
# for line in f:
#     id, imdbid, _ = line.split(",")
#     ia = imdb.IMDb()
#     m = ia.get_movie(imdbid)
#     info = ia.get_movie_main(m.getID())
#     keywords = ia.get_movie_keywords(m.getID())
#     if 'data' in keywords and "keywords" in keywords["data"]:
#         info["data"]["keywords"] = keywords["data"]["keywords"]
#     info_set[int(id)] = info
#     print id

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
    features_f = theano.misc.pkl_utils.load(fs.get(db.fs.files.find_one({"dataset": "MovieLens"})["_id"]))
    return model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
           popular_l,  cold_l, features_f

model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l, \
popular_l,  cold_l, features_f  = load_data()


def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True, dynamic=True)
    save("MovieLens", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, subject_thres=5, min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)


for n_factors in (70, 40, 10):
    model = VisualOffsetBPR(n_factors, n_users_l, n_items_l, features_f,
                            lambda_u=0.1, lambda_v=0.1, lambda_bias=1,
                            lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=3, width=256,
                            margin=1.0, learning_rate=.05, lambda_v_off=0.1,
                            embedding_rescale=0.1, batch_size=200000)

    train_and_save(model)

for n_factors in (40, 10):
    model = VisualFactorKBPR(n_factors, n_users_l, n_items_l, features_f,
                             lambda_u=0, lambda_v=0, lambda_bias=0.1, lambda_variance=10, max_norm=1.0, lambda_cov=0,
                             lambda_weight_l1=0, lambda_weight_l2=0.0, dropout_rate=0.5, n_layers=3, width=256,
                             margin=.5, learning_rate=.05,  lambda_v_off=.1,
                             embedding_rescale=0.1, batch_size=200000, warp_count=20)

    train_and_save(model)

for n_factors in (100, 10, 40, 70):
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0.0, lambda_v=0, loss="bpr")
    train_and_save(model)
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=0.0, lambda_v=1E-5, loss="bpr")
    train_and_save(model)
    model = LightFMModel(n_factors, n_users_l, n_items_l, lambda_u=1E-6, lambda_v=1E-6, loss="bpr")
    train_and_save(model)



def train_and_save(model):
    model = early_stop(model, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
                      patience=5, validation_frequency=1, n_epochs=10000000, adagrad=True, dynamic=True)
    save("MovieLens", model, n_users_l, n_items_l, train_dict_l, valid_dict_l, test_dict_l, exclude_dict_l, cold_dict_l,
         popular_l,  cold_l, subject_thres=5, min_rating=4.0, user_rating_count=10, tag_freq_thres=20, use_director=False)
train_and_save(model)
for n_factors in (100,):
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
