from utils import *
from pymongo import MongoClient
import random
from pandas import *
db = MongoClient().models
key_set = set()
for m in db["fs.files"].find():
    key_set = key_set.union(set(m.keys()))

data = {}
keys = sorted(key_set)
for key in keys:
    data[key] = []
for m in db["fs.files"].find():
    line = []
    for key in keys:
        if key in m:
            data[key].append(m[key])
        else:
            data[key].append(None)
df = DataFrame(data=data)
df.groupby(['class', 'K', 'n_factors']).apply




