from utils import *
from model import Popularity, UserKNN
from pymongo import MongoClient
import random
db = MongoClient().models

# for dataset in MongoClient().datasets["fs.files"].find():
#     if db["fs.files"].find({"dataset_id": dataset["_id"],
#                             "class": "UserKNN",
#                             "test_recall_100": {"$exists":True}
#                             }).count() == 0:
#         train, valid, test, exclude = load_dataset(dataset["_id"])
#         if db["fs.files"].find({"dataset_id": dataset["_id"]}).count() > 0:
#             record = list(db["fs.files"].find({"dataset_id": dataset["_id"]}))[0]
#             print record
#             model = UserKNN(record["n_users"], record["n_items"], 200)
#             model.train(train)
#             save(record["dataset"], model, record["n_users"], record["n_items"],
#                  train, valid, test, exclude, None, None, None, _id=record["dataset_id"])
#     #

# for record in db["fs.files"].find({"class": "LightFMModel"}):
#     model = load_model(record["_id"])[0]
#     record["with_features"] = model.V_features_orig is not None
#     print record["with_features"]
#     db["fs.files"].save(record)
#     print "Save" + str(model)
while True:
    try:
        for record in list(db["fs.files"].find({"dataset": "MovieLens",
                                                "test_recall_100_90p_30000s": {"$exists":False}
                                                })):
            try:
                eval_one_for_cold_items(record["_id"])
            except Exception as e:
                print
    except Exception as e:
        print e

