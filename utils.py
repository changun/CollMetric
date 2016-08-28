import numpy
import time
import sys
import json
import numpy as np
import pickle
import gzip
import urllib2
from collections import defaultdict

from pymongo import MongoClient, DESCENDING
from isbnlib import clean

client = MongoClient()


def early_stopping(model, train_dict, valid_dict, exclude_dict, metric_fn, pre="", save_model=True, n_epochs=1000,
                   validation_frequency=50, patience=1000, valid_per_user_sample=50, start_adagrad=300,
                   start_hard_case=300, **kwargs):
    sys.stderr.write("Model %s n_epochs %d\n" % (model, n_epochs))
    sys.stderr.flush()
    status = []
    # early-stopping parameters
    # look as this many epochs
    patience_increase = 300  # wait this much longer when a new best is  found
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
    best_metric = numpy.Infinity
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

            best_metric = min(best_metric, metric_fn(model))
            this_validation_loss = -valid_auc
            train_auc, train_auc_sem = model.validate(None, train_dict, per_user_sample=valid_per_user_sample)
            epoch += validation_frequency
            status.append([epoch, valid_auc, train_auc, valid_auc_sem, train_auc_sem, cur_train_loss])
            sys.stderr.write(
                "\rEpoch: %d time: %d best: %f cur %f" % (epoch, time.time() - start, best_train_loss, cur_train_loss))
            sys.stderr.flush()
            # print model.precision(valid_dict, train_dict, n=10)[0]
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
            with open(pre + str(model) + ".p", "wb") as f:
                pickle.dump(copy, f)
            print ("Save: " + pre + str(model) + ".p")
        return best_metric
    return best_metric


def early_stop(model, train_dict_bc, metric_fn, n_epochs=2000, validation_frequency=100, patience=300, **kwargs):
    import cStringIO, numpy
    import theano.misc.pkl_utils
    output = None
    best_metric = numpy.Infinity

    try:
        patience_count = patience
        print model
        while True:
            model.train(train_dict_bc, epoch=validation_frequency,  **kwargs)

            new_metric = metric_fn(model)
            if new_metric > best_metric:
                # worse than the best metric
                patience_count -= validation_frequency
                print ("Patience %d" % (patience_count,))
                if "f" in model.__dict__ and model.learning_rate > 0.01:
                     model.learning_rate = 0.01
                     print "New learning rate %g" % (model.learning_rate,)
                     model.f = None

            else:


                # the best metric
                # recover patience
                patience_count = patience
                # record best metric
                best_metric = new_metric
                # take a snapshot
                output = cStringIO.StringIO()
                theano.misc.pkl_utils.dump(model, output)
                output.flush()
            n_epochs -= validation_frequency
            if patience_count <= 0 or n_epochs <= 0:
                raise Exception()
    except Exception as e:
        print e
    finally:
        if output is not None:
            best_model = theano.misc.pkl_utils.load(output)
            return best_metric, best_model
        return None


def data_to_dict(data):
    data_dict = defaultdict(set)
    items = set()
    for (user, item) in data:
        data_dict[user].add(item)
        items.add(item)
    return data_dict, set(data_dict.keys()), items


def split(user_dict, portion, popular_item_percentile, seed=1):
    numpy.random.seed(seed)
    items_count = defaultdict(int)
    for u, items in user_dict.items():
        for i in items:
            items_count[i] += 1

    train_dict = {}
    test_dict = {}
    valid_dict = {}
    exclude_dict = {}
    cold_item_test_dict = {}
    cold_item_thres = -numpy.partition(-numpy.asarray(items_count.values()), 100)[100]
    all_items = set(items_count.keys())
    cold_items = set([i for i, c in items_count.items() if c < cold_item_thres])
    popular_items = all_items - cold_items

    for user, items in user_dict.iteritems():
        # for in_matrix items
        if len(items) > 0:
            chunk_size = len(items) // sum(portion)
            if chunk_size > 0:
                items = list(items)
                numpy.random.shuffle(items)
                test_dict[user] = set(items[0:chunk_size * portion[2]])
                valid_dict[user] = set(items[chunk_size * portion[2]: chunk_size * (portion[2] + portion[1])])
                train_dict[user] = set(
                    items[chunk_size * (portion[2] + portion[1]): chunk_size * (portion[2] + portion[1] + portion[0])])
                exclude_dict[user] = set(items[chunk_size * (portion[2] + portion[1]):])
                if len(test_dict[user] & cold_items) > 0:
                    cold_item_test_dict[user] = test_dict[user] & cold_items
            else:
                train_dict[user] = items
                exclude_dict[user] = items

    print("Seed %d, Portion %s/%s, Popular/Cold %d/%d Train %d, Valid %d, Test %d, Exclude %d Cold %d" %
          (seed, portion, popular_item_percentile,
           len(popular_items), len(cold_items),
           sum([len(items) for items in train_dict.values()]),
           sum([len(items) for items in valid_dict.values()]),
           sum([len(items) for items in test_dict.values()]),
           sum([len(items) for items in exclude_dict.values()]),
           sum([len(items) for items in cold_item_test_dict.values()])

           ))
    return train_dict, valid_dict, test_dict, exclude_dict, cold_item_test_dict, popular_items, cold_items


def preprocess(user_dict, portion, popular_item_percentile=90, seed=1):
    train_dict, valid_dict, test_dict, exclude_dict, cold_item_dict, popular_items, cold_items = split(user_dict,
                                                                                                       portion=portion,
                                                                                                       popular_item_percentile=popular_item_percentile,
                                                                                                       seed=seed)
    n_items = max(popular_items | cold_items) + 1
    n_users = max(user_dict.keys()) + 1
    print("n_items %d n_users %d\n" % (n_items, n_users))
    return n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict, cold_item_dict, popular_items, cold_items


def v_features(photo_ids, path="/data/dataset/features_all.npy"):
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


def flickr(max_likes=100, features=False):
    user_dict, photo_ids, user_ids = pickle.load(open("/data/dataset/dat_100_100000.p", "rb"))
    numpy.random.seed(1)
    user_dict_reduced = dict(
        [(i_index, set(numpy.random.choice(list(items), size=min(len(items), max_likes), replace=False))) for
         i_index, items
         in
         user_dict.iteritems()])
    if features:
        V_features = v_features(photo_ids)
        return user_dict_reduced, V_features, dict([(id, photo) for photo, id in photo_ids.items()])
    else:
        return user_dict_reduced

def flickr_details(id_photos):
    def details(id):
        return {
            "title": id_photos[id],
            "img": "http://flickr-images.s3-website-us-east-1.amazonaws.com/"+id_photos[id]+".jpg"}
    return details

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


def lastfm(tag_freq_thres=20, user_thres=50, first_tag_only=False, include_play_count=False):
    import pymongo
    import sklearn.preprocessing
    user_dict = {}
    user_ids = {}
    item_ids = {}
    coll = pymongo.MongoClient().lastfm.artist
    available_items = set(coll.distinct("_id"))
    user_count = defaultdict(int)
    with open("/data/dataset/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv") as f:
        for l in f.readlines():
            fields = l.split("\t")
            if len(fields) > 3:
                username = fields[0].strip()
                itemname = fields[1].strip()
                if itemname in available_items:
                    user_count[username] += 1
    with open("/data/dataset/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv") as f:
        for l in f.readlines():
            fields = l.split("\t")
            if len(fields) > 3:
                username = fields[0].strip()
                itemname = fields[1].strip()
                if itemname in available_items and user_count[username] >= user_thres:
                    if username not in user_ids:
                        user_ids[username] = len(user_ids)
                        user_dict[user_ids[username]] = set()
                    if itemname not in item_ids:
                        item_ids[itemname] = len(item_ids)
                    user_dict[user_ids[username]].add(item_ids[itemname])
        feature_dict = {}
        listeners = []
        playcount = []
        to_title = {}
        for item, id in item_ids.items():
            artist = coll.find_one(item, ["artist.tags.tag.name", "artist.stats.listeners", "artist.stats.playcount", "artist.name"])
            to_title[id] = artist["artist"]["name"]
            try:
                if first_tag_only:
                    feature_dict[id] = [tag["name"].strip().lower() for tag in artist["artist"]["tags"]["tag"]][:1]
                else:
                    feature_dict[id] = [tag["name"].strip().lower() for tag in artist["artist"]["tags"]["tag"]]

            except KeyError:
                pass
            if "listeners" in artist["artist"]["stats"]:
                listeners.append(int(artist["artist"]["stats"]["listeners"]))
            else:
                listeners.append(0)
            if "playcount" in artist["artist"]["stats"]:
                playcount.append(int(artist["artist"]["stats"]["playcount"]))
            else:
                playcount.append(0)

    features, labels = feature_sets_to_array(feature_dict, tag_freq_thres, len(item_ids))
    if include_play_count:
        listeners = sklearn.preprocessing.scale(listeners)
        playcount = sklearn.preprocessing.scale(playcount)

        features = numpy.hstack((features,
                                 listeners.reshape((len(listeners), 1)),
                                 playcount.reshape((len(playcount), 1)),
                                 ))
        labels += ["listeners", "playcount"]

    return user_dict, features, labels, to_title

def lastfm_details(to_title):
    def details(id):
        return {"title": to_title[id],
                "url": "http://www.last.fm/music/" + to_title[id]}
    return details



def import_lastfm_tags():
    import urllib, json, pyprind, time

    mbids = set()

    with open("userid-timestamp-artid-artname-traid-traname.tsv") as f:
        for l in f:
            fields = l.split("\t")
            if len(fields) > 3:
                mbids.add((fields[2], fields[3]))
    with open("/data/dataset/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv") as f:
        for l in f:
            fields = l.split("\t")
            if len(fields) > 3:
                mbids.add((fields[1], fields[2]))
    db = client.lastfm
    bar = pyprind.ProgBar(len(mbids))
    for mbid, name in mbids:
        bar.update(1, mbid)
        if db.artist.find({"_id": mbid}).count() > 0 or db.artist.find({"_id": name}).count() > 0 or len(mbid) == 0:
            continue
        try:
            url = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&api_key=84e85abb3f7f44f9b67bc7f3031e3be6&format=json&mbid=" + mbid
            response = urllib.urlopen(url)
            data = json.loads(response.read())
            data["_id"] = mbid
            if "artist" not in data:
                url = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&api_key=84e85abb3f7f44f9b67bc7f3031e3be6&format=json&artist=" + name
                response = urllib.urlopen(url)
                data = json.loads(response.read())
                data["_id"] = name

            if "artist" in data:
                db.artist.save(data)
            else:
                print "Error: " + mbid + " " + url
            time.sleep(0.2)

        except Exception as e:
            print e
            print "Error:" + mbid


def feature_sets_to_array(feature_dict, thres, n_items):
    from scipy.sparse import coo_matrix
    feature_freq = defaultdict(int)
    for features in feature_dict.values():
        for f in features:
            feature_freq[f] += 1

    feature_to_dim = dict(
        [(feature, dim) for dim, feature in enumerate([f for f in feature_freq.keys() if feature_freq[f] >= thres])])
    rows = []
    cols = []
    val = []
    for id, features in feature_dict.items():
        for feature in features:
            if feature in feature_to_dim:
                rows.append(id)
                cols.append(feature_to_dim[feature])
                val.append(1)
    return coo_matrix((val, (rows, cols)), shape=(n_items, len(feature_to_dim))), sorted(feature_to_dim.keys(),
                                                                                         key=lambda f: feature_to_dim[
                                                                                             f])


def movielens10M(min_rating=0.0, tag_freq_thres=20):
    from StringIO import StringIO
    import sklearn.preprocessing
    from zipfile import ZipFile
    from urllib import urlopen
    from scipy.sparse import csc_matrix
    import pymongo
    import re

    url = urlopen("/data/dataset/ml-10m.zip")
    user_dict = {}
    user_ids = {}
    item_ids = {}
    with ZipFile(StringIO(url.read())) as f:
        for l in f.open("ml-10M100K/ratings.dat"):
            user, item, rating, time = l.split("::")
            user = int(user)
            item = int(item)
            rating = float(rating)
            if rating >= min_rating:
                time = long(time)
                if user not in user_ids:
                    user_ids[user] = len(user_ids)
                    user_dict[user_ids[user]] = set()
                if item not in item_ids:
                    item_ids[item] = len(item_ids)
                time = long(time)
                user_dict[user_ids[user]].add(item_ids[item])
    feature_dict = defaultdict(set)
    years = []
    coll = pymongo.MongoClient().movielens.imdb
    url = urlopen("/data/dataset/ml-10m.zip")
    with ZipFile(StringIO(url.read())) as f:
        for l in f.open("ml-10M100K/movies.dat"):
            item, name, genre = l.split("::")
            if int(item) in item_ids:
                item_id = item_ids[int(item)]
                imdb_record = coll.find_one(int(item), ["year", "keywords", "director"])
                if imdb_record is not None:
                    if "keywords" in imdb_record["keywords"]["data"]:
                        keywords = imdb_record["keywords"]["data"]["keywords"]
                    else:
                        keywords = []
                    # directors = map(lambda d: d["personID"], imdb_record["director"])

                    year = imdb_record["year"]
                else:
                    keywords = []
                    year = int(re.search('\((\d+)\)', l).group(1))

                years.append(year)
                feature_dict[item_id] = [g.lower().strip() for g in genre.split("|")] + keywords
        features, labels = feature_sets_to_array(feature_dict, tag_freq_thres, len(item_ids))
        years = sklearn.preprocessing.scale(years)
        features = csc_matrix(numpy.hstack((features.toarray(), years.reshape((len(years), 1)))))
    return user_dict, features, labels


def movielens20M(min_rating=3.0, user_rating_count=50, tag_freq_thres=20, use_director=False):
    import sklearn.preprocessing
    from zipfile import ZipFile
    from scipy.sparse import csc_matrix
    import pymongo
    import re
    import csv

    user_dict = {}
    user_ids = {}
    item_ids = {}
    with ZipFile(open("/data/dataset/ml-20m.zip")) as f:
        ls = csv.reader(f.open("ml-20m/ratings.csv"), delimiter=',', quotechar='"')
        ls.next()
        user_count = defaultdict(int)
        for user, item, rating, time in ls:
            user = int(user)
            rating = float(rating)
            if rating >= min_rating:
                user_count[user] += 1

        ls = csv.reader(f.open("ml-20m/ratings.csv"), delimiter=',', quotechar='"')
        ls.next()
        for user, item, rating, time in ls:
            user = int(user)
            item = int(item)
            rating = float(rating)
            if user_count[user] >= user_rating_count and rating >= min_rating:
                time = long(time)
                if user not in user_ids:
                    user_ids[user] = len(user_ids)
                    user_dict[user_ids[user]] = set()
                if item not in item_ids:
                    item_ids[item] = len(item_ids)
                time = long(time)
                user_dict[user_ids[user]].add(item_ids[item])
        ls = csv.reader(f.open("ml-20m/movies.csv"), delimiter=',', quotechar='"')
        ls.next()
        id_to_title = {}
        for raw_item_id, title, genere in ls:
            item = int(raw_item_id)
            if item in item_ids:
                id_to_title[item_ids[item]] = title
        feature_dict = defaultdict(set)
        years = []
        director_dict = defaultdict(set)
        coll = pymongo.MongoClient().movielens.imdb
        ls = csv.reader(f.open("ml-20m/movies.csv"), delimiter=',', quotechar='"')
        ls.next()
        for item, name, genre in ls:
            if int(item) in item_ids:
                item_id = item_ids[int(item)]
                imdb_record = coll.find_one(int(item), ["year", "keywords", "director.personID", "title"])
                keywords = []
                if imdb_record is not None and "keywords" in imdb_record["keywords"]["data"]:
                    keywords = imdb_record["keywords"]["data"]["keywords"]
                    if "director" in imdb_record:
                        director_dict[item_id] = [d["personID"] for d in imdb_record["director"]]
                year_re = re.search('\((\d+)\)', name)
                if year_re is not None:
                    year = int(year_re.group(1))
                else:
                    year = 1984

                feature_dict[item_id] = [g.lower().strip() for g in genre.split("|")] + keywords + [
                    "year-" + str((year / 5) * 5)]

    features, labels = feature_sets_to_array(feature_dict, tag_freq_thres, len(item_ids))
    director, director_id = feature_sets_to_array(director_dict, 3, len(item_ids))
    if use_director:
        labels += director_id
        features = numpy.hstack((features, director))

    return user_dict, csc_matrix(features), labels, id_to_title


def movielens20M_genere(min_rating=3.0, user_rating_count=50, tag_freq_thres=20):
    import sklearn.preprocessing
    from zipfile import ZipFile
    from scipy.sparse import csc_matrix
    import pymongo
    import re
    import csv

    user_dict = {}
    user_ids = {}
    item_ids = {}
    with ZipFile(open("/data/dataset/ml-20m.zip")) as f:
        ls = csv.reader(f.open("ml-20m/ratings.csv"), delimiter=',', quotechar='"')
        ls.next()
        user_count = defaultdict(int)
        for user, item, rating, time in ls:
            user = int(user)
            rating = float(rating)
            if rating >= min_rating:
                user_count[user] += 1

        ls = csv.reader(f.open("ml-20m/ratings.csv"), delimiter=',', quotechar='"')
        ls.next()
        for user, item, rating, time in ls:
            user = int(user)
            item = int(item)
            rating = float(rating)
            if user_count[user] >= user_rating_count and rating >= min_rating:
                time = long(time)
                if user not in user_ids:
                    user_ids[user] = len(user_ids)
                    user_dict[user_ids[user]] = set()
                if item not in item_ids:
                    item_ids[item] = len(item_ids)
                time = long(time)
                user_dict[user_ids[user]].add(item_ids[item])
        ls = csv.reader(f.open("ml-20m/movies.csv"), delimiter=',', quotechar='"')
        ls.next()
        id_to_title = {}
        for raw_item_id, title, genere in ls:
            item = int(raw_item_id)
            if item in item_ids:
                id_to_title[item_ids[item]] = title
        feature_dict = defaultdict(set)
        ls = csv.reader(f.open("ml-20m/movies.csv"), delimiter=',', quotechar='"')
        ls.next()
        for item, name, genre in ls:
            if int(item) in item_ids:
                item_id = item_ids[int(item)]
                year_re = re.search('\((\d+)\)', name)
                if year_re is not None:
                    year = int(year_re.group(1))
                else:
                    year = 1984

                feature_dict[item_id] = [g.lower().strip() for g in genre.split("|")] + [
                    "year-" + str((year / 5) * 5)]

    features, labels = feature_sets_to_array(feature_dict, tag_freq_thres, len(item_ids))
    return csc_matrix(features)




def nextflix(min_rating=3.0, user_freq_thres=50, n_users=None):
    from os import listdir
    from os.path import isfile, join
    path = "/data/dataset/download/training_set"
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    user_count = defaultdict(int)
    user_movie = []
    for f in files:
        f = open(f)
        mid = int(f.next()[0:-2]) - 1
        for line in f:
            user, rating, time = line.split(",")
            user = int(user)
            rating = float(rating)
            if rating >= min_rating:
                user_movie.append((user, mid))
                user_count[user] += 1
    users = set([u for u, c in user_count.items() if c>=user_freq_thres])
    if n_users != None:
        users = set(numpy.random.choice(list(users), size=n_users, replace=False))
    user_ids = {}
    user_dict = {}
    for user, mid in user_movie:
        if user in users:
            if user not in user_ids:
                uid = len(user_ids)
                user_ids[user] = uid
                user_dict[uid] = list()
            else:
                uid = user_ids[user]
            user_dict[uid].append(mid)
    for u in user_dict:
        user_dict[u] = set(user_dict[u])
    return user_dict

def neflix_title(with_year = True):
    ret = {}
    for l in open("/data/dataset/download/movie_titles.txt"):
        l = l.strip()
        try:
            eles = l.split(",")
            id = eles[0]
            year = eles[1]
            name = ",".join(eles[2:])
            if with_year:
                ret[int(id)-1] = "(" + year + ")" + name
            else:
                ret[int(id) - 1] = name
        except Exception as e:
            pass
    return ret

def netflix_details():
    import cPickle
    titles = neflix_title()
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))

    def details(id):
        #movie = cPickle.load(gzip.open('/data/imdb/' + netflix_imdb[id + 1] + ".p", 'rb'))
        return  {"title": titles[id],
                 "url": "http://www.imdb.com/title/tt" + netflix_imdb[id+1] if id+1 in netflix_imdb else None}
    return details




def isbn_to_work(isbn_10):
    isbn_10 = clean(isbn_10)

    def merge_two_dicts(x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z
    try:
        books = list(client.bookcx.edition.find({"isbn_10": isbn_10}).limit(1))
        if len(books) > 0:
            book = books[0]
            subjects = book["subjects"] if "subjects" in book else []
            authors = map(lambda a: a["key"], book["authors"]) if "authors" in book else []

            if "works" in book and len(book["works"]) > 0:
                work_id = book["works"][0]["key"].split("/")[-1]
                work = client.bookcx.work.find({"_id.id": work_id}).sort([["_id.rev", DESCENDING]]).limit(1).next()
                subjects = (work["subjects"] if "subjects" in work else []) + subjects
                authors =  (map( lambda a: a["author"]["key"] , work["authors"]) if "authors" in work else []) + authors
                book = merge_two_dicts(work, book)

            book["authors"] = set(authors)
            normalized_subjects = []
            for subject in subjects:
                if type(subject) == dict:
                    normalized_subjects.append(subject["key"].split("/")[-1].lower())
                else:
                    normalized_subjects.append(subject.lower())
            book["subjects"] = normalized_subjects
            return book
    except StopIteration:
        return None
    except Exception as e:
        print e
        raise e


def bookcx_features():
    import pyprind
    item_dict = {}
    with open("BX-Books.csv") as f:
        count = len(f.readlines()) - 1
    bar = pyprind.ProgBar(count)
    with open("BX-Books.csv") as f:
        f.readline()
        for l in f.readlines():
            isbn = l.split(";")[0]
            item_dict[isbn] = isbn_to_work(isbn)
            bar.update(1, isbn)
    return item_dict


def bookcx(item_thres=5, user_thres=5, subject_thres=20):
    import re
    # these tags are not relevant to the books themselves
    exclude_set = {u"protected daisy", u"accessible book", u"in library", u"overdrive"}
    user_dict = {}
    item_dict = {}
    with open("BX-Book-Ratings.csv") as f:
        f.readline()
        for l in f.readlines():
            fields = l.split(";")
            if len(fields) == 3:
                username = fields[0]
                itemname = fields[1]
                if username not in user_dict:
                    user_dict[username] = set()
                if itemname not in item_dict:
                    item_dict[itemname] = set()
                item_dict[itemname].add(username)
                user_dict[username].add(itemname)
    user_ids = dict(
        [(u, index) for index, u in enumerate([u for u, items in user_dict.items() if len(items) >= user_thres])])
    item_ids = dict(
        [(i, index) for index, i in enumerate([i for i, users in item_dict.items() if len(users) >= item_thres])])
    user_dict = {}

    with open("BX-Book-Ratings.csv") as f:
        f.readline()
        for l in f.readlines():
            fields = l.split(";")
            if len(fields) == 3:
                username = fields[0]
                itemname = fields[1]
                if username in user_ids and itemname in item_ids:
                    if user_ids[username] not in user_dict:
                        user_dict[user_ids[username]] = set()
                    user_dict[user_ids[username]].add(item_ids[itemname])
    features_dict = {}
    titles = [""] * len(item_ids)
    for isbn, id in item_ids.items():
        work = isbn_to_work(isbn)
        titles[id] = ("(%s) %s" % (isbn,  work["title"] if work is not None and "title" in work else "")).strip()
        # extract subjects/authors
        features = set()
        if work is not None and "subjects" in work and len(work["subjects"]) > 0:
            for s in work["subjects"]:
                for real_s in re.split('[,/-]+', s):
                    real_s = real_s.strip().lower()
                    if len(real_s) > 0 and real_s not in exclude_set:
                        features.add(real_s)
        if work is not None and "authors" in work:
            for author in work["authors"]:
                features.add(author)
        features_dict[id] = features

    features, labels = feature_sets_to_array(features_dict, subject_thres, n_items=len(item_ids))

    return user_dict, features, labels, titles

def book_details_fn(titles):
    id_isbn = dict([[id, title[2:12]] for id, title in enumerate(titles)])
    def details(id):
        work = isbn_to_work(id_isbn[id])
        if work is None:
            return {"title": id_isbn[id]}
        else:
            return {"title": work["title"], "url": "https://openlibrary.org" + work["key"]}
    return details




def import_openlibrary():
    import ujson
    import pyprind
    bar = pyprind.ProgBar(49601384)
    db = client.bookcx
    types = {"author", "edition", "work"}
    with open("/mnt/ol_dump_latest.txt") as f:
        for l in f:
            type, id, rev, dt, dat = l.split("\t")
            type = type.split("/")[2]
            if type in types:
                dat = ujson.loads(dat)
                id = id.split("/")[-1]
                dat["_id"] = {"id": id, "rev": int(rev)}
                db[type].save(dat, check_keys=False)
            bar.update(1, item_id=type)


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


def diversity(model, test_dict, exclude_dict, features, n=100, n_users=3000, exclude_items=None, fn="cosine",
              hit_only=True, div_groud_truth=True):
    import scipy.spatial.distance
    numpy.random.seed(1)
    users = numpy.random.choice(test_dict.keys(), size=min(n_users, len(test_dict)), replace=False, )
    tops = model.topN(users, exclude_dict, n=n, exclude_items=exclude_items)
    diversity = []
    cache = {}
    distance_func = eval("scipy.spatial.distance." + fn)

    def average_similarity(items):
        similarity = 0
        count = 0
        for i in items:
            for j in items:
                if i > j:
                    count += 1
                    if (i, j) in cache:
                        similarity += cache[(i, j)]
                    else:
                        if (features[i] ** 2).sum() > 0 and (features[j] ** 2).sum() > 0:
                            distance = distance_func(features[i], features[j])
                            similarity += distance
                            cache[(i, j)] = distance
                        else:
                            similarity += 1
                            cache[(i, j)] = 1
        return similarity / float(count)

    for i in range(len(users)):
        if hit_only:
            recommendations = list(set(tops.next()) & test_dict[users[i]])
        else:
            recommendations = tops.next()
        if len(recommendations) > 1 and len(test_dict[users[i]]) > 0:
            top_diversity = average_similarity(recommendations)
            if div_groud_truth:
                ground_truth_diversity = average_similarity(test_dict[users[i]])
                diversity.append(top_diversity / ground_truth_diversity)
            else:
                diversity.append(top_diversity)
        else:
            diversity.append(0)
    return numpy.mean(diversity)


def diversity_user(model, test_dict, exclude_dict, user_dict, n=100, n_users=3000, exclude_items=None, normalize=True,
                   hit_only=True):
    item_dict = defaultdict(set)
    for u, items in user_dict.items():
        for i in items:
            item_dict[i].add(u)

    numpy.random.seed(1)
    users = numpy.random.choice(test_dict.keys(), size=min(n_users, len(test_dict)), replace=False, )
    tops = model.topN(users, exclude_dict, n=n, exclude_items=exclude_items)
    diversity = []
    cache = {}

    def average_similarity(items):
        similarity = 0
        count = 0
        for i in items:
            for j in items:
                if i != j and i > j:
                    count += 1
                    if (i, j) not in cache:
                        cache[(i, j)] = 1 - (len(item_dict[i] & item_dict[j]) / float(len(item_dict[i] | item_dict[j])))
                    similarity += cache[(i, j)]
        return similarity / float(count)

    for i in range(len(users)):
        if hit_only:
            recommendations = list(set(tops.next()) & test_dict[users[i]])
        else:
            recommendations = tops.next()
        if len(recommendations) > 1 and len(test_dict[users[i]]) > 1:
            top_diversity = average_similarity(recommendations)
            if normalize:
                ground_truth_diversity = average_similarity(test_dict[users[i]])
                diversity.append(top_diversity / ground_truth_diversity)
            else:
                diversity.append(top_diversity)
        else:
            diversity.append(0)
    return numpy.mean(diversity)


def to_dense_user_item_factor(n_factor, features, train_dict, n_users):
    from sklearn.decomposition import NMF
    from scipy.sparse import csc_matrix, lil_matrix
    features = csc_matrix(features)
    item_factors = NMF(n_factor).fit_transform(features)
    user_features_count = lil_matrix((n_users, features.shape[1]))
    for u, items in train_dict.items():
        user_features_count[u] = features[list(items)].sum(0)
    user_factors = NMF(n_factor).fit_transform(user_features_count)
    return item_factors, user_factors


def metrics(m, users, test_dict, exclude_dict, n=100, exclude_items=None):
    # return top N item (sorted)
    mean_ranks = []
    mrr = []
    ndcg = []
    for inx, scores in enumerate(m.scores_for_users(users)):
        user = users[inx]
        if exclude_dict is not None and user in exclude_dict:
            scores[list(exclude_dict[user])] = -numpy.Infinity
        if exclude_items is not None and len(exclude_items) > 0:
            scores[exclude_items] = -numpy.Infinity
        ranks = numpy.argsort(numpy.argsort(-scores))[list(test_dict[user])]
        mrr.append(1.0 / (numpy.min(ranks)+1))
        ndcg.append(numpy.sum(1.0/ numpy.log2(ranks + 2)) / numpy.sum(1.0/ numpy.log2(numpy.arange(len(test_dict[user])) + 2)))

    return numpy.mean(mrr), numpy.mean(ndcg)


def closest_movie(m, id, to_title):
    import theano
    print ("Most like to %d %s" % (id, to_title[id]))
    print "\n".join(map(lambda id: str((id,to_title[id])), theano.function([], ((m.V[id] - m.V)**2).sum(1))().argsort()[0:30]))

def user_movie(m, uid, user_dict, to_title, exclude_items=None, exclude_dict=None, n=20):
    import theano
    import theano.tensor as T

    tops = m.topN([uid], exclude_dict, n=n, exclude_items=exclude_items).next()
    print "Hit %d items" % (len(set(tops) & user_dict[uid]))
    U = m.U.get_value().reshape((m.K, m.n_users, m.n_factors))
    for i in range(m.K):
        for j in range(m.K):
            if j > i:
                print (i, j, ((U[i, uid] - U[j, uid])**2).sum())
    i = T.lvector()
    j = T.lvector()
    clusters = list(theano.function([i, j], m.assign_cluster(i, j))([uid] * len(tops), tops) / m.n_users)
    cluster_set = defaultdict(list)
    for i, item_id in enumerate(tops):
        cluster_set[clusters[i]].append(item_id)
    for cluster, items in cluster_set.items():
        print "Cluster %d (Density %g) Recommend %d Hit %d" % (cluster, m.mixture_density.get_value()[cluster, uid], len(items), len(set(items) & user_dict[uid]))
        for item in items:
            if item in user_dict[uid]:
                hit = "\x1b[31m**\x1b[0m"
            else:
                hit = "  "

            print "%s %s %s" % (hit, item, to_title[item])
        print "\n\n"




def closest_movie_fm(m, id, to_title):
    import theano
    print ("Most like to %d %s" % (id, to_title[id]))
    print "\n".join(map(lambda id: str((id,to_title[id])),  (-numpy.inner(m.model.item_embeddings[id],m.model.item_embeddings)).argsort()[0:30]))



def closest_movie_embedding(m, id, to_title):
    import theano

    print ("Most like to %d %s" % (id, to_title[id]))
    V = m.V_embedding(range(m.n_items))
    print "\n".join(map(lambda id: str((id,to_title[id])), theano.function([], ((V[id] - V)**2).sum(1))().argsort()[0:30]))


def closest_movie_by_content(m, id, to_title):
    print ("Most like to %d %s" % (id, to_title[id]))
    print "\n".join(map(lambda id: str((id, to_title[id])),
                        (-numpy.inner(m[id], m)).argsort()[0:10]))


def taste_kid():
    from purl import URL
    from pyquery import PyQuery as pq
    import urllib3
    from collections import defaultdict
    import time
    user_titles = defaultdict(set)

    for uid in range(231474+1):
        for t in ["s", "m", "b" "h", "g"]:
            url = "http://www.tastekid.com/profile/resources/%d/1/%s/added/0/1000" % (uid, t)
            while True:
                http = urllib3.PoolManager()
                r = http.request('GET', url)
                if len(r.data) == 0:
                    break
                try:
                    d = pq(r.data)
                    for div in d("div.tk-Resource"):
                        user_titles[uid].add((t, div.attrib["title"]))
                    buttons = d("button")
                    if len(buttons) > 0:
                        url = "http://www.tastekid.com/" + buttons[0].attrib["data-endpoint"]
                    else:
                        break
                except Exception as e:
                    print e
                    print uid, r.status, r.data
                    break
            #time.sleep(1)
        print "%d users data" % (len(user_titles))

def kmeans(U, V, train_dict, k):
    def find_centers(X, k):
        import sklearn.cluster
        c = sklearn.cluster.KMeans(k, n_jobs=-1)
        assignments = c.fit_predict(X)

        return c.cluster_centers_
    U_k = numpy.zeros((k, U.shape[0],U.shape[1],))
    for u in range(len(U)):
        print u
        items = list(train_dict[u])
        if len(items) > k *5 :
            U_k[:, u,:] = find_centers(V[items], k)
        else:
            U_k[:, u, :] = U[u]
    return U_k



def gap_statistic(X, test=5):
    def Wk(mu, clusters):
        K = len(mu)
        return sum([np.linalg.norm(mu[i] - c) ** 2 / (2 * len(c)) \
                    for i in range(K) for c in clusters[i]])

    def bounding_box(X):
        min = numpy.min(X, axis=0)
        max = numpy.max(X, axis=0)
        return min, max

    min, max = bounding_box(X)
    print min, max
    def find_centers(X, k):
        import sklearn.cluster
        c = sklearn.cluster.KMeans(k, n_jobs=-1)
        assignments = c.fit_predict(X)
        clusters = defaultdict(list)
        for i,a in enumerate(assignments):
            clusters[a].append(X[i])

        return c.cluster_centers_, clusters

    # Dispersion for real distribution
    ks = range(1, test)
    Wks = numpy.zeros(len(ks))
    Wkbs = numpy.zeros(len(ks))
    sk = numpy.zeros(len(ks))
    max_distance = numpy.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X, k)
        for i in range(len(mu)):
            for j in range(len(mu)):
                if j > i:
                    max_distance[indk] = numpy.max([max_distance[indk], np.linalg.norm(mu[i] - mu[j])])
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 5
        BWkbs = numpy.zeros(B)
        for i in range(B):
            Xb = []
            for col in range(len(min)):
                Xb.append(numpy.random.uniform(min[col], max[col], size=len(X)))
            Xb = np.array(Xb).T
            if not Xb.shape == X.shape:
                print Xb.shape, X.shape
                return
            mu, clusters = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs) / B
        sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk]) ** 2) / B)
    sk = sk * np.sqrt(1 + 1 / B)
    gaps = Wkbs - Wks
    determine = []
    for i in range(0, test-2):
        determine.append(gaps[i] > (gaps[i+1] + sk[i+1]))
    return (ks, Wks, Wkbs, sk, gaps, determine,max_distance)


def model_selection(models, valid, test, exclude, n=10):
    import pyprind
    def hit(predictions, targets):
        return reduce(lambda hits, p: hits + (p in targets), predictions, 0)
    users = test.keys()
    tops = []
    for m in models:
        tops.append(m.topN(users, exclude, n=n, exclude_items=None))
    valids = []
    for u in users:
        valids.append(map(lambda model: hit(model.next(), valid[u]), tops))
    selections = numpy.argmax(valids, axis=1)
    total_recall = 0
    for i in range(len(models)):
        users_for_model = [u for u, s in zip(users, selections) if s == i ]
        total_recall += models[i].recall(test, merge_dict(valid, exclude),  users=users_for_model, ns=[n])[0][0] * len(users_for_model)

    return total_recall / len(users)

def mixture_density(m, u, train):
    import theano
    import theano.tensor as T
    clusters = T.lvector()
    items = T.lvector()
    clusters_i = T.repeat(clusters, items.shape[0])
    weights = T.exp(m.factor_score_cluster_ij(clusters_i, T.concatenate([items] * m.K)).reshape(
        (items.shape[0], m.K)).T)  # (K, items)
    weights /= weights.sum(0)  # (K, items)
    f = theano.function([clusters, items], weights.sum(1) / weights.sum())
    items = train[u]
    return f(([u + (i * m.n_users) for i in range(m.K)]), list(items))

def medium(thres=10):
    import json
    user_dict = {}
    user_counts = defaultdict(int)
    user_ids = {}
    doc_ids = {}
    user_doc = json.load(open("/data/dataset/user-doc.json"))
    articles = json.load(open("/data/dataset/articles.json"))
    doc_titles = dict([(a['id'], a['title']) for a in articles])
    article_tags = dict([(a['id'], a['tags']) for a in articles])
    titles = []
    feature_dict = defaultdict(set)
    for u, d in user_doc:
        user_counts[u] += 1
    for u, d in user_doc:
        if user_counts[u] >= thres and user_counts[u]<3000:
            if u not in user_ids:
                user_ids[u] = len(user_ids)
                user_dict[user_ids[u]] = set()
            if d not in doc_ids:
                doc_ids[d] = len(doc_ids)
                titles.append(d + "/" + doc_titles[d])
                for tag in article_tags[d]:
                    feature_dict[doc_ids[d]].add(tag["name"])
            user_dict[user_ids[u]].add(doc_ids[d])
    features, labels = feature_sets_to_array(feature_dict, 10, n_items=len(doc_ids))
    return user_dict, features, labels, titles


def medium_details(titles):
    def details(id):
        return {"title": titles[id].split("/")[1],
                "url": "https://medium.com/articles/" + titles[id].split("/")[0]}
    return details

# for i in numpy.random.randint(n_users_l, size=50):
#     if len(list(train_dict_l[i])) > 20:
#         print i, gap_statistic(m_f.V.get_value()[list(train_dict_l[i])])
# import gc
# out = []
# for n_factor in [10, 20, 30, 40]:
#     for i in range(10):
#         numpy.random.seed()
#         cov = 10 ** numpy.random.uniform(-1, 2)
#         bias = 10 ** numpy.random.uniform(-1, 1)
#         variance = 10 ** numpy.random.uniform(-1, 2)
#         gc.collect()
#         gc.collect()
#         model = KBPRModel(n_factor, n_users_n, n_items_n,
#                               batch_size=100000, lambda_v=0.0, lambda_u=0.0,
#                                per_user_sample=20,
#                                learning_rate=0.1,
#                                variance_mu=1.0, uneven_sample=True,
#                                update_mu=True,
#                                lambda_variance=variance,
#                                warp=20, max_norm=1.1, K=1, lambda_bias=bias, lambda_cov=cov, margin=.5, lambda_density=0.0001,
#                                bias_range=(1E-6, 10))
#         out.append(early_stop(model, train_dict_n, lambda m: -m.recall(valid_dict_n, train_dict_n, n_users=3000)[0][0],
#                    patience=1000, validation_frequency=250, n_epochs=10000000, adagrad=True))
#         numpy.random.seed()
#         v = 10 ** numpy.random.uniform(-10, -4)
#         fm = LightFMModel(n_factor, n_users_n, n_items_n, lambda_v=v)
#         out.append(early_stop(fm, train_dict_n, lambda m: -m.recall(valid_dict_n, train_dict_n, n_users=3000)[0][0],
#                               patience=300, validation_frequency=50, n_epochs=10000000, adagrad=True))
#
#
#
# out = []
# for factors in [50, 20, 10]:
#     for bias, var in [[0.8, 0.5], [0.5, 0.5], [0.1, 0.5], [0.5, 0.3], [0.5, 0.1]]:
#         lastfm_50 = KBPRModel(factors, n_users_l, n_items_l, batch_size=100000, lambda_v=0.0, lambda_u=0.0,
#                               per_user_sample=20,
#                               learning_rate=0.1,
#                               variance_mu=1.0, uneven_sample=True,
#                               update_mu=True,
#                               lambda_variance=var,
#                               warp=20, max_norm=1.1, K=1, lambda_bias=bias, lambda_cov=1, margin=.5, lambda_density=0.0001,
#                               bias_range=(1E-6, 10))
#         out.append(early_stop(lastfm_50, train_dict_l, lambda m: -m.recall(valid_dict_l, train_dict_l, n_users=3000)[0][0],
#                    patience=2000, validation_frequency=150, n_epochs=10000000, adagrad=True))
# out = []
# for i in range(10):
#     u = 10 ** -numpy.random.uniform(3, 8)
#     medium_fm = LightFMModel(50, n_users_m, n_items_m, lambda_v=u, lambda_u=u)
#     out.append(early_stop(medium_fm, train_dict_m, lambda m: -m.recall(valid_dict_m, train_dict_m, n_users=3000)[0][0],
#                patience=200, validation_frequency=50, n_epochs=10000000, adagrad=True))


def lsh(dataset, distance='euclidean_squared', lsh_family="cross_polytope", n_hash_bits=18, n_rotations=1, n_tables=50):
    # or negative_inner_product
    import falconn
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = lsh_family
    params_cp.distance_function = distance
    params_cp.l = n_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = n_rotations
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = 'bit_packed_flat_hash_table'
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(n_hash_bits, params_cp)
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    return table

def concate(fm, type="bias"):
    V = numpy.concatenate((fm.model.item_embeddings, fm.model.item_biases.reshape((fm.model.item_embeddings.shape[0], 1))), axis=1)
    U = numpy.concatenate((fm.model.user_embeddings, numpy.ones((fm.model.user_embeddings.shape[0], 1), dtype="float32")), axis=1)
    U /=  ((U ** 2).sum(1) ** 0.5).reshape((U.shape[0], 1))
    V /=  ((V ** 2).sum(1) ** 0.5).max()
    if type == "bias":
        pass
    elif type == "simple":
        V = numpy.concatenate((V, ((1 - (V**2).sum(1))**0.5).reshape((fm.model.item_embeddings.shape[0], 1))), axis=1)
        U = numpy.concatenate(
            (U, numpy.zeros((fm.model.user_embeddings.shape[0], 1), dtype="float32")), axis=1)
    elif type == "L2":
        V = V * 0.83
        V_norm = (V ** 2).sum(1) ** 0.5
        for m in range(3):
            V = numpy.concatenate((V, (V_norm ** ((m+1)*2)).reshape((fm.model.item_embeddings.shape[0], 1))),
                                  axis=1)
            U = numpy.concatenate(
                (U, numpy.zeros((fm.model.user_embeddings.shape[0], 1), dtype="float32")+ 0.5), axis=1)
    return U, V

def lsh_recall(U, V, test, valid, exclude, n=10,
               distance='euclidean_squared',
               lsh_family="cross_polytope",
               n_hash_bits=18, n_rotations=1,  n_tables=50, max_accuracy=0.14,
               mu=None, bias=None, oversample=5):
    table = lsh(V, distance, lsh_family, n_hash_bits, n_rotations, n_tables)
    table.set_num_probes(800)
    # number_of_tables = n_tables
    # def evaluate_number_of_probes(number_of_probes):
    #     table.set_num_probes(number_of_probes)
    #     numpy.random.seed(1)
    #     recalls = []
    #     for u in numpy.random.choice(test.keys(), size=10000):
    #         #exclude_items = exclude[u]
    #         tops = table.find_k_nearest_neighbors(U[u], n)
    #         real_tops = numpy.argpartition(((U[u] - V) ** 2).sum(1), n)[0:n]
    #         recall = len(set(tops) & set(real_tops)) / float(n)
    #         recalls.append(recall)
    #     return numpy.mean(recalls)
    # number_of_probes = number_of_tables
    # while True:
    #     accuracy = evaluate_number_of_probes(number_of_probes)
    #     print('{} -> {}'.format(number_of_probes, accuracy))
    #     if accuracy >= 0.9 * max_accuracy:
    #         break
    #     number_of_probes = number_of_probes * 2
    #
    # if number_of_probes > number_of_tables:
    #     left = number_of_probes // 2
    #     right = number_of_probes
    #     while right - left > 1:
    #         number_of_probes = (left + right) // 2
    #         accuracy = evaluate_number_of_probes(number_of_probes)
    #         print('{} -> {}'.format(number_of_probes, accuracy))
    #         if accuracy >= 0.9 * max_accuracy:
    #             right = number_of_probes
    #         else:
    #             left = number_of_probes
    #     number_of_probes = right
    # print('Done')
    # print('{} probes'.format(number_of_probes))
    exclude = merge_dict(exclude, valid)
    recalls = []
    if mu is not None:
        mu = 2* (mu ** 2)
        bias = numpy.log(bias)
    total_time = 0
    for u in test.keys():
        exclude_items = exclude[u]
        start = time.time()
        tops = table.find_k_nearest_neighbors(U[u], (n*oversample) + len(exclude_items))
        if mu is None:
            items = numpy.asarray(filter(lambda item: item not in exclude_items, tops))
            scores = -(U[u] * V[items]).sum(1)
            tops = items[numpy.argpartition(scores, n)[0:n]]
        else:
            items = numpy.asarray(filter(lambda item: item not in exclude_items, tops))

            scores = ((U[u] - V[items]) ** 2).sum(1) / mu[u] - bias[items]
            tops = items[numpy.argpartition(scores, n)[0:n]]

        total_time += time.time() - start
        recall = len(set(tops) & test[u]) / float(len(test[u]))
        recalls.append(recall)
    print "Time:" + str(total_time / len(test.keys()))
    print numpy.mean(recalls)
