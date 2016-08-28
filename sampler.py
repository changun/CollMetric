import pyximport

pyximport.install()
import numpy
from multiprocessing import Pool
import fast_utils
import sys
import signal
import gc

_cur_samplers = {}


def _sample(sid):
    global _cur_samplers
    return _cur_samplers[sid].next()


def _del(sid):
    gc.collect()
    global _cur_samplers
    del _cur_samplers[sid]
    print ("%d samplers left" % (len(_cur_samplers), ))

def _batch_sampler(sampler, batch_size, n_users, n_items):
    users = numpy.arange(n_users)
    items = numpy.arange(n_items)

    def permuted_triplet():
        while True:
            full_triplet = sampler.next()
            while full_triplet.shape[0] < batch_size:
                full_triplet = numpy.concatenate((full_triplet, sampler.next()))
            numpy.random.shuffle(full_triplet)
            yield full_triplet

    gen = permuted_triplet()
    triplet = gen.next()
    index = 0
    while True:
        end_index = index + batch_size
        if end_index <= len(triplet):
            sub_triplet = triplet[index:end_index]
            index = end_index
        else:
            next_triplet = gen.next()
            sub_triplet = numpy.concatenate([triplet[index:], next_triplet[0:end_index - len(triplet)]])
            index = end_index - len(triplet)
            triplet = next_triplet
        if sub_triplet.shape[0] != batch_size:
            raise Exception("Error in Batch sampler. sub_triplet shape:" + str(sub_triplet.shape))
        yield sub_triplet


def _sample_uneven(train_dict, exclude_items, n_items):
    users = []
    pos_items = []
    for u, items in train_dict.items():
        for i in items:
            users.append(u)
            pos_items.append(i)
    triplet = numpy.zeros((len(users), 4), "int64")
    triplet[:, 0] = users
    triplet[:, 1] = pos_items
    triplet[:, 3] = numpy.arange(len(users))
    # we will reuse this array
    while True:
        # additional item samples other than that already in unique_pos_j
        available_items = numpy.asarray([i for i in range(n_items) if i not in exclude_items])
        triplet[:, 2] = numpy.random.choice(available_items, size=len(users))
        for i in xrange(len(users)):
            if triplet[i, 2] in train_dict[triplet[i, 0]]:
                triplet[i, 2] = numpy.random.randint(n_items)
                while triplet[i, 2] in train_dict[triplet[i, 0]] or triplet[i, 2] in exclude_items:
                    triplet[i, 2] = numpy.random.randint(n_items)
        yield triplet


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


def _init():
    # it is important to reseed the random generator in each process
    numpy.random.seed()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigterm_handler)

def _sample_warp(train_dict, exclude_items, n_items, batch_size, warp_count):
    import gc, random
    from scipy.sparse import dok_matrix
    from utils import dict_to_coo
    pos_sample_count = sum([len(items) for items in train_dict.values()])
    pairs = numpy.zeros((pos_sample_count, 2), dtype="int64")
    index = 0


    for u, items in train_dict.items():
        for i in items:
            pairs[index][0] = u
            pairs[index][1] = i
            index += 1
    numpy.random.seed()
    numpy.random.shuffle(pairs)
    index = 0
    sampled_pairs = numpy.zeros((batch_size*warp_count, 3), dtype="int64")
    gc.collect()
    negative_samples = [-1] * (warp_count * batch_size)
    random.seed()
    while True:
        if index+batch_size > pairs.shape[0]:
            numpy.random.seed()
            numpy.random.shuffle(pairs)
            index = 0
            gc.collect()
        sampled_pos_pairs = pairs[index:index + batch_size]
        random.seed()
        sample_index = 0
        for u, _ in sampled_pos_pairs:
            pos_items = train_dict[u]
            for _ in xrange(warp_count):
                j_neg = random.randint(0, n_items-1)
                while j_neg in pos_items:
                    j_neg = random.randint(0, n_items-1)
                negative_samples[sample_index] = j_neg
                sample_index += 1
        sampled_pairs[:, 0:2] = numpy.repeat(sampled_pos_pairs, warp_count, axis=0)
        sampled_pairs[:, 2] = negative_samples
        index = index + batch_size
        yield sampled_pairs


def _init_sampler(train_dict, exclude_dict, exclude_items, n_items, per_user_sample, batch_size=-1, uneven=False, warp=False):
    n_users = max(train_dict.keys()) + 1
    sid = len(_cur_samplers)
    if warp is not False and warp > 0:
        _cur_samplers[sid] = _sample_warp(train_dict, exclude_dict, n_items, batch_size, warp)
        return sid
    if uneven:
        basic_sampler = _sample_uneven(train_dict, exclude_items, n_items)
    else:
        basic_sampler = fast_utils.sample(train_dict, exclude_dict, exclude_items, n_items, per_user_sample)
    if batch_size > 0:
        _cur_samplers[sid] = _batch_sampler(basic_sampler, batch_size, n_users, n_items)
    else:
        _cur_samplers[sid] = basic_sampler
    return sid


_pools = []
import multiprocessing
for i in range(int(multiprocessing.cpu_count() * 1.5)):
    _pool = Pool(processes=1)
    _pool.apply_async(_init)
    _pools.append(_pool)


def _create_sampler_real(train_dict, exclude_dict, n_items, per_user_sample, batch_size, uneven, exclude_items=None, warp=False):
    if exclude_items is None:
        exclude_items = set()
    else:
        exclude_items = set(exclude_items)

    rets = []
    sid = None
    for _pool in _pools:
        sid = _pool.apply(_init_sampler,
                          (train_dict, exclude_dict, exclude_items, n_items, per_user_sample, batch_size, uneven, warp))

        rets.append(_pool.apply_async(_sample, (sid,)))
    i = 0
    try:
        while True:
            res = rets[i % len(_pools)]
            samples = res.get()
            rets[i % len(_pools)] = _pools[i % len(_pools)].apply_async(_sample, (sid,))
            i += 1
            yield samples
    finally:
        print ("Being GCed sid %d" % (sid, ))
        for _pool in _pools:
            _pool.apply_async(_del, (sid,))



sampler_cache = {}
def create_sampler(train_dict, exclude_dict, n_items, per_user_sample, batch_size, uneven, exclude_items=None, warp=False):
    if exclude_dict is None:
        exclude_dict = {}
    if exclude_items is None:
        exclude_items = set()
    key = (sum(map(lambda u: sum(u[1])%(u[0]+1), train_dict.items())),
           sum(map(lambda u: sum(u[1])%(u[0]+1), exclude_dict.items())),
           n_items,
           per_user_sample,
           batch_size,
           uneven,
           warp,
           sum(exclude_items))
    if key in sampler_cache:
        try:
            if len(sampler_cache[key].next()) > 0:
                print "Use cached sampler"
                return sampler_cache[key]
        except Exception:
            pass

    sampler_cache[key] = _create_sampler_real(train_dict, exclude_dict, n_items, per_user_sample, batch_size, uneven, exclude_items, warp)
    return sampler_cache[key]

def clean_sampler():
    sampler_cache = {}