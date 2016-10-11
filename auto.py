from utils import *
from model import Popularity
from pymongo import MongoClient
import random
db = MongoClient().models

# for dataset in MongoClient().datasets["fs.files"].find():
#     train, valid, test, exclude = load_dataset(dataset["_id"])
#     if db["fs.files"].find({"dataset_id": dataset["_id"]}).count() > 0:
#         record = list(db["fs.files"].find({"dataset_id": dataset["_id"]}))[0]
#         print record
#         model = Popularity(record["n_users"], record["n_items"])
#         model.train(train)
#         save(record["dataset"], model, record["n_users"], record["n_items"],
#              train, valid, test, exclude, None, None, None, _id=dataset["_id"])


# for record in db["fs.files"].find({"class": "LightFMModel"}):
#     model = load_model(record["_id"])[0]
#     record["with_features"] = model.V_features_orig is not None
#     print record["with_features"]
#     db["fs.files"].save(record)
#     print "Save" + str(model)
# while True:
#     try:
#         records = db["fs.files"].find({"valid_recall_100": {"$exists": False}})
#         if records.count() > 0:
#             record = records[random.randint(0, records.count()-1)]
#             eval_one(record["_id"])
#         else:
#             break
#     except Exception as e:
#         print e

while True:

    try:
        records = db["fs.files"].find({"test_recall_100_90p_30000s": {"$exists": False}})
        if records.count() > 0:
            record = records[random.randint(0, records.count()-1)]
            eval_one_for_cold_items(record["_id"])
        else:
            break
    except Exception as e:
        print e
