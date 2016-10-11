from utils import *
from pymongo import MongoClient
from itertools import groupby
db = MongoClient().models
for key, group in groupby(db["fs.files"].find({"dataset": "Citeulike"}), lambda r: [r["class"], r["n_factors"]]):
    print key
model, train, valid, test, exclude = load_model(id)

model_record = db["fs.files"].find_one(id)

for key, group in groupby(db["fs.files"].find({"dataset": "Medium"}), lambda r: str(r["class"])):
    max_recall = 0
    max_name = None
    for r in group:
        max_recall = max(max_recall, r["test_recall_100"])
        if max_recall == r["test_recall_100"]:
            max_name = r["name"]
    print max_name, max_recall
