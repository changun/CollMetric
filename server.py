from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import threading
import re
from collections import defaultdict
import json
import theano
import theano.tensor as T


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise Exception()


def get_user_details(model, uid, n=100):
    ret = {}
    m = model["model"]
    item_details_fn = model["item_details"]
    train = model["train"][uid] if uid in model["train"] else set()
    test = model["test"][uid] if uid in model["test"] else set()
    ret["train"] = train
    ret["test"] = test
    ret["n"] = n
    ret["uid"] = uid
    ret["n_clusters"] = m.K
    ret["densities"] = tuple([float(f) for f in m.mixture_density.get_value()[range(m.K), uid]])
    print ret["densities"]
    tops = m.topN([uid], model["train"], n=n).next()
    # compute distances between mixtures
    U = m.U.get_value().reshape((m.K, m.n_users, m.n_factors))
    ret["distances"] = []
    for i in range(m.K):
        for j in range(m.K):
            if j > i:
                ret["distances"].append((i, j, ((U[i, uid] - U[j, uid]) ** 2).sum() ** 0.5))

    i = T.lvector()
    j = T.lvector()
    tops_and_train = tops + list(train)
    clusters_assignments_fn = theano.function([i, j], m.assign_cluster(i, j))
    clusters_assignments = list(clusters_assignments_fn([uid] * len(tops_and_train),
                                                        tops_and_train) / m.n_users)

    cluster_recommendation = defaultdict(list)
    cluster_train = defaultdict(list)
    for i, item_id in enumerate(tops):
        cluster_recommendation[clusters_assignments[i]].append([item_id, item_id in test])
    for i, item_id in enumerate(list(train)):
        cluster_train[clusters_assignments[i + len(tops)]].append([item_id])
    ret["cluster_train"] = cluster_train
    ret["cluster_recommendation"] = cluster_recommendation
    ret["item_details"] = dict([(item, item_details_fn(item)) for item in tops_and_train])
    return ret


class LocalData(object):
    models = {}


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if None != re.search('/api/.+/user_details/\d+', self.path):
            model = self.path.split('/')[-3]
            user = int(self.path.split('/')[-1])
            if LocalData.models.has_key(model):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(get_user_details(LocalData.models[model], user), default=set_default))
        elif None != re.search('/api/.*/item_details/\d+', self.path):
            model = self.path.split('/')[-3]
            item = int(self.path.split('/')[-1])
            if LocalData.models.has_key(model):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(LocalData.models[model]["item_details"](item), default=set_default))
        elif None != re.search('/demo/', self.path):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = open("index.html")
            for l in html:
                self.wfile.write(l)
        else:
            self.send_response(403)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

    def shutdown(self):
        self.socket.close()
        HTTPServer.shutdown(self)


class SimpleHttpServer():
    def __init__(self, ip, port):
        self.server = ThreadedHTTPServer((ip, port), HTTPRequestHandler)

    def start(self):
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def waitForThread(self):
        self.server_thread.join()

    def addModel(self, name, model, item_details, train, test):
        LocalData.models[name] = {"model": model,
                                  "item_details": item_details,
                                  "train": train,
                                  "test": test}

    def stop(self):
        self.server.shutdown()
        self.waitForThread()
