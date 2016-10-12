from utils import *
from model import Popularity
from pymongo import MongoClient
import random
db = MongoClient().models

# for dataset in MongoClient().datasets["fs.files"].find()[0:1]:
#     train, valid, test, exclude = load_dataset(dataset["_id"])
#     if db["fs.files"].find({"dataset_id": dataset["_id"]}).count() > 0:
#         record = list(db["fs.files"].find({"dataset_id": dataset["_id"]}))[0]
#         print record
#         model = Popularity(record["n_users"], record["n_items"])
#         model.train(train)
#         save(record["dataset"], model, record["n_users"], record["n_items"],
#              train, valid, test, exclude, None, None, **dataset)
#

# for record in db["fs.files"].find({"class": "LightFMModel"}):
#     model = load_model(record["_id"])[0]
#     record["with_features"] = model.V_features_orig is not None
#     print record["with_features"]
#     db["fs.files"].save(record)
#     print "Save" + str(model)
while True:
    try:
        records = db["fs.files"].find({"test_recall_100_95p_30000s": numpy.NaN,

                                       "_id": {"$in":[-5407812687832705599,
 -6303119507362106509,
 -8418818630298044899,
 -6674105543508680440,
 4897553002029964742,
 1352375330236313303,
 5098886450963647303,
 5061537450437342293,
 6217472712540135861,
 -7402463773975596579,
 2555559890983899977,
 -7272793283626686020,
 -4188698187257419937,
 8306548813031573954,
 -6847242372680792284,
 -7954141949836316174,
 -6598785277516796820,
 -2275105200429561392,
 -8559308637579230101,
 -393068671623664627,
 -7629517226456837874,
 -3195446459950145227,
 1771933032492821707,
 -5734531796512145287,
 3474066032554430762,
 -594901706749496652,
 4536545879026272129,
 4983764759617874322]}
                                     })
        if records.count() > 0:
            record = records[random.randint(0, records.count()-1)]
            eval_one(record["_id"])
    except Exception as e:
        print e

