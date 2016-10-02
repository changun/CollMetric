from model import *
from utils import *
import theano.misc.pkl_utils



def train_fn(model, train, valid):
    return early_stop(model, train, lambda m: -m.recall(valid, train, n_users=3000)[0][0],
                      patience=500, validation_frequency=100, n_epochs=10000000, adagrad=True)[1]


def save(method, model, name):
    import os, shutil
    results = []
    try:
        results = theano.misc.pkl_utils.load(open(name, "rb"))
    except Exception:
        pass

    results.append({"method": method, "model": model})
    tmp = os.tempnam()
    theano.misc.pkl_utils.dump(results, open(tmp, "wb"))
    shutil.move(tmp, name)
