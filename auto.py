from utils import *
from pymongo import MongoClient
db = MongoClient().models
while True:
    records = db["fs.files"].find({"valid_recall_100": {'$exists': False}, "class": {'$ne': "LightFMModel"}})
    if records.count() > 0:
        eval_one(records[0]["_id"])