import numpy
import time
import sys
import json
import numpy as np
import pickle
import gzip
import urllib2
from collections import defaultdict


def early_stopping(model, train_dict, valid_dict, exclude_dict, pre="", save_model=True, n_epochs=1000,
                   validation_frequency=10, patience=200, valid_per_user_sample=50, start_adagrad=300,
                   start_hard_case=300, **kwargs):
    sys.stderr.write("Model %s n_epochs %d\n" % (model, n_epochs))
    sys.stderr.flush()
    status = []
    # early-stopping parameters
    # look as this many epochs
    patience_increase = 200  # wait this much longer when a new best is  found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    # go through this many
    # epochs before checking the model

    best_params = None
    best_validation_loss = numpy.inf
    best_train_loss = numpy.inf
    patience_for_train_loss = 2
    done_looping = False

    epoch = 0
    copy = None
    try:
        while (epoch < n_epochs) and (not done_looping):

            start = time.time()
            adagrad = epoch >= start_adagrad
            hard_case = epoch >= start_hard_case
            if adagrad:
                sys.stderr.write(
                    "\n Adagrad kick in")
                sys.stderr.flush()
            if hard_case:
                sys.stderr.write(
                    "\n Hardcase in")
                sys.stderr.flush()
            cur_train_loss = model.train(train_dict, epoch=validation_frequency, adagrad=adagrad, hard_case=hard_case,
                                         **kwargs)

            valid_auc, valid_auc_sem = model.validate(train_dict, valid_dict, per_user_sample=valid_per_user_sample)
            this_validation_loss = -valid_auc
            train_auc, train_auc_sem = model.validate(None, train_dict, per_user_sample=valid_per_user_sample)
            epoch += validation_frequency
            status.append([epoch, valid_auc, train_auc, valid_auc_sem, train_auc_sem, cur_train_loss])
            sys.stderr.write(
                "\rEpoch: %d time: %d best: %f cur %f" % (epoch, time.time() - start, best_train_loss, cur_train_loss))
            sys.stderr.flush()
            #print model.precision(valid_dict, train_dict, n=10)[0]
            # check train loss
            if cur_train_loss > best_train_loss:
                if patience_for_train_loss == 0:
                    sys.stderr.write(
                        "\nEpoch: %d Train Loss Increase! best: %f cur %f\n" % (epoch, best_train_loss, cur_train_loss))
                    sys.stderr.flush()
                else:
                    patience_for_train_loss -= 1
            else:
                best_train_loss = cur_train_loss
                patience_for_train_loss = 2

            # check validation loss
            if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, epoch + patience_increase)
                # store the best model so far!
                if save_model:
                    copy = model.copy()
                    copy["status"] = status

                best_validation_loss = this_validation_loss

                sys.stderr.write(
                    "\nEpoch: %d New best valid loss(auc) %f sem %f train loss %f train auc %f sem %f patience %d\n" % (
                        epoch, best_validation_loss, valid_auc_sem, cur_train_loss, train_auc, train_auc_sem, patience))
                sys.stderr.flush()
            else:
                sys.stderr.write(
                    "\nEpoch: %d Overfit!!?? valid loss(auc) %f best valid loss(auc) %f train loss %f train auc %f sem %f patience %d\n" % (
                        epoch, this_validation_loss, best_validation_loss, cur_train_loss, train_auc, train_auc_sem,
                        patience))
                sys.stderr.flush()

            if patience <= epoch:
                done_looping = True
                break
    finally:
        if copy is not None:
            with open(pre + str(model) +".p", "wb") as f :
                pickle.dump(copy, f)
            print ("Save: " + pre + str(model) +".p")
    return model.recall(valid_dict, train_dict)[0]


def data_to_dict(data):
    data_dict = defaultdict(set)
    items = set()
    for (user, item) in data:
        data_dict[user].add(item)
        items.add(item)
    return data_dict, set(data_dict.keys()), items


def split(user_dict, portion, seed=1):
    numpy.random.seed(seed)
    train_dict = {}
    test_dict = {}
    valid_dict = {}
    exclude_dict = {}
    all_items = set()
    for user, items in user_dict.iteritems():
        all_items |= items
        chunk_size = len(items) // sum(portion)
        if chunk_size > 0:
            items = list(items)
            numpy.random.shuffle(items)
            test_dict[user] = set(items[0:chunk_size * portion[2]])
            valid_dict[user] = set(items[chunk_size * portion[2]: chunk_size * (portion[2] + portion[1])])
            train_dict[user] = set(
                items[chunk_size * (portion[2] + portion[1]): chunk_size * (portion[2] + portion[1] + portion[0])])
            exclude_dict[user] = set(items[chunk_size * (portion[2] + portion[1]):])
        else:
            train_dict[user] = items
            exclude_dict[user] = items

    print("Seed %d, Portion %s, Train %d, Valid %d, Test %d, Exclude %d" %
          (seed, portion,
           sum([len(items) for items in train_dict.values()]),
           sum([len(items) for items in valid_dict.values()]),
           sum([len(items) for items in test_dict.values()]),
           sum([len(items) for items in exclude_dict.values()])))
    return train_dict, valid_dict, test_dict, exclude_dict, all_items


def split_for_user_cold_start(user_dict, valid_hold_out_user_ids, test_hold_out_user_ids, n_likes, max_n_likes=5):
    numpy.random.seed(1)
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    all_items = set()
    for u, items in user_dict.iteritems():
        all_items |= items
        if u in valid_hold_out_user_ids:
            items = list(items)
            numpy.random.shuffle(items)
            if n_likes > 0:
                train_dict[u] = set(items[:n_likes])
            valid_dict[u] = set(items[max_n_likes:])
        elif u in test_hold_out_user_ids:
            items = list(items)
            numpy.random.shuffle(items)
            if n_likes > 0:
                train_dict[u] = set(items[:n_likes])
            test_dict[u] = set(items[max_n_likes:])
        else:
            train_dict[u] = items
    n_items = max(all_items) + 1
    n_users = len(user_dict)
    return n_items, n_users, train_dict, valid_dict, test_dict


def preprocess(user_dict, portion, seed=1):
    train_dict, valid_dict, test_dict, exclude_dict, all_items = split(user_dict, portion=portion, seed=seed)
    n_items = max(all_items) + 1
    n_users = max(user_dict.keys()) + 1
    print("n_items %d n_users %d\n" % (n_items, n_users))
    return n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict


def v_features(photo_ids, path="/home/ubuntu/dataset/features_all.npy"):
    id_raw_id = {}
    names = json.loads(urllib2.urlopen('https://s3-us-west-1.amazonaws.com/cornell-nyc-sdl-tmp/images.json').read())
    index = 0
    for name in [name.split(".")[0] for name in names]:
        if name in photo_ids:
            id_raw_id[photo_ids[name]] = index
        index += 1
    all_features = numpy.load(path, mmap_mode='r')
    return all_features[[id_raw_id[i] for i in range(len(photo_ids))], :]


def u_features(user_ids, user_list="users_list.pickle", features="users_feature.npy"):
    ul = pickle.load(open(user_list, "rb"))
    raw_users_feature = numpy.load(features)

    U_features = numpy.zeros((len(user_ids.keys()), 1024)) + 0.001
    for inx, u in enumerate(ul):
        if u in user_ids:
            U_features[user_ids[u], :] = raw_users_feature[inx, :]
    return U_features


def merge_dict(a, b):
    c = defaultdict(set)
    for key, items in a.iteritems():
        c[key] |= items
    for key, items in b.iteritems():
        c[key] |= items
    return c


def sample():
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    f = gzip.open('flickrUserPhoto.csv.gz', 'rb')
    while True:
        line = f.readline()
        if not line:
            break
        user, photo = line[:-1].split(",")
        user_count[user] += 1
        item_count[photo] += 1
    names = dict([(name.split(".")[0], index) for index, name in enumerate(
        json.loads(urllib2.urlopen('https://s3-us-west-1.amazonaws.com/cornell-nyc-sdl-tmp/images.json').read()))])
    f = gzip.open('flickrUserPhoto.csv.gz', 'rb')
    user_ids = {}
    photo_ids = {}
    user_dict = defaultdict(set)
    while True:
        line = f.readline()
        if not line:
            break
        user, photo = line[:-1].split(",")
        if user_count[user] >= 25 and item_count[photo] >= 10 and item_count[photo] <= 200 and photo in names:
            if user not in user_ids:
                user_ids[user] = len(user_ids)
            if photo not in photo_ids:
                photo_ids[photo] = len(photo_ids)

            user_dict[user_ids[user]].add(photo_ids[photo])
    return user_dict, photo_ids, user_ids


def sample_cold_items(photo_ids, user_ids, path='/mnt/flickrUserPhoto.csv.gz'):
    names = dict([(name.split(".")[0], index) for
                  index, name in enumerate(
            json.loads(urllib2.urlopen('https://s3-us-west-1.amazonaws.com/cornell-nyc-sdl-tmp/images.json').read()))])

    cold_user_dict = defaultdict(set)
    with gzip.open(path, 'rb') as f:
        lines = (line.strip().split(",") for line in f)
        # get all user items pairs not in the given data
        user_items = [[user, item] for user, item in lines if
                      item in names and item not in photo_ids and user in user_ids]
        # sample n cold items
        numpy.random.seed(1)
        cold_items = numpy.random.choice(list(set([item for user, item in user_items])), replace=False, size=100000)
        # create cold_item_to_id dict
        cold_item_ids = dict([(item, inx) for inx, item in enumerate(cold_items)])
        # get all user_cold_item pairs
        cold_user_items = ((user, item) for user, item in user_items if item in cold_item_ids)
        # create user_to_cold_items dict
        for user, item in cold_user_items:
            cold_user_dict[user_ids[user]].add(cold_item_ids[item])
        # get cold items' features
        features = v_features(cold_item_ids)
    return cold_user_dict, cold_item_ids, features


def sample_holdout_user(user_dict, user_ids):
    user_own = json.loads(urllib2.urlopen('https://s3.amazonaws.com/flickr-images/user-own.json').read())
    user_own_count = defaultdict(int)
    for u, i in user_own:
        user_own_count[u] += 1
    id_users = dict([[id, user] for user, id in user_ids.iteritems()])
    numpy.random.seed = 1
    pickle.dump(numpy.random.choice([id_users[user] for user, items in user_dict.iteritems() if
                                     len(items) >= 5 and user_own_count[id_users[user]] >= 10], size=10000,
                                    replace=False).tolist()
                , open("hold_out_users.p", "wb"))





def movielens1M(min_rating=0.0):
    from StringIO import StringIO
    from zipfile import ZipFile
    from urllib import urlopen
    url = urlopen("http://files.grouplens.org/datasets/movielens/ml-1m.zip")
    user_dict = {}
    with ZipFile(StringIO(url.read())) as f:
        for l in f.open("ml-1m/ratings.dat").readlines():
            user, item, rating, time = l.split("::")
            user = int(user)
            item = int(item)
            rating = float(rating)
            time = long(time)
            if rating >= min_rating:
                if user not in user_dict:
                    user_dict[user] = set()
                user_dict[user].add(item)
    return user_dict

def lastfm():
    user_dict = {}
    user_ids = {}
    item_ids = {}
    with open("userid-timestamp-artid-artname-traid-traname.tsv") as f:
        for l in f.readlines():
            fields = l.split("\t")
            if len(fields) > 3:
                username = fields[0]
                itemname = fields[2]
                if username not in user_ids:
                    user_ids[username] = len(user_ids)
                    user_dict[user_ids[username]] = set()
                if itemname not in item_ids:
                    item_ids[itemname] = len(item_ids)
                user_dict[user_ids[username]].add(item_ids[itemname])
    return user_dict




def movielens10M(min_rating=0.0):
    from StringIO import StringIO
    from zipfile import ZipFile
    from urllib import urlopen
    url = urlopen("http://files.grouplens.org/datasets/movielens/ml-10m.zip")
    user_dict = {}
    with ZipFile(StringIO(url.read())) as f:
        for l in f.open("ml-10M100K/ratings.dat").readlines():
            user, item, rating, time = l.split("::")
            user = int(user)
            item = int(item)
            rating = float(rating)
            time = long(time)
            if rating >= min_rating:
                if user not in user_dict:
                    user_dict[user] = set()
                user_dict[user].add(item)
    return user_dict



def coo_to_dict(m):
    dat = {}
    rows, cols = m.nonzero()
    for i, u in enumerate(rows):
        if u not in dat:
            dat[u] = set()
        dat[u].add(cols[i])
    return dat


def dict_to_coo(d, n_rows, n_cols):
    from scipy.sparse import coo_matrix
    data = []
    row = []
    col = []
    for u, items in d.items():
        for i in items:
            data.append(1)
            row.append(u)
            col.append(i)
    return coo_matrix((numpy.asarray(data), (numpy.asarray(row), numpy.asarray(col))), shape=(n_rows, n_cols))


