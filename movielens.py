#!c:\Python2.4\python.exe
try:
    from imdb import IMDb
    from utils import *
    import os, sys, time
    import imdb
    import cPickle
    import re
    import gzip, cPickle
    import pyprind

except ImportError:
    print 'You need to install the IMDbPY package!'
    sys.exit(1)


def import_imdb():
    i = IMDb()
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))
    no_year = neflix_title(with_year=False)
    for id, title in neflix_title().items():
        netflix_id = id + 1
        if netflix_id not in netflix_imdb:
            try:
                ms = i.search_movie(title)
                if len(ms) > 0:
                    m = ms[0]
                else:
                    ms = i.search_movie(no_year[id])
                    if len(ms) > 0:
                        m = ms[0]
                    else:
                        short_name = re.split("[:\(/]+|(Season \d)|(Trilogy)", no_year[id])[0]
                        print "%s -> %s" % (no_year[id], short_name)
                        m = i.search_movie(short_name)[0]

                imdb_id = m.movieID

                movie = i.get_movie(m.movieID)
                movie["keywords"] = i.get_movie_keywords(m.movieID)["data"]
                #movie["votes_details"] = i.get_movie_vote_details(imdb_id)["data"]
                #movie["critic_reviews"] = i.get_movie_critic_reviews(imdb_id)["data"]
                #movie["awards"] = i.get_movie_awards(imdb_id)["data"]
                movie["business"] = i.get_movie_business(imdb_id)["data"]
                #movie["recommendations"] = i.get_movie_recommendations(imdb_id)["data"]
                movie["synopsis"] = i.get_movie_synopsis(imdb_id)["data"]
                #movie["news"] = i.get_movie_news(imdb_id)
                
                netflix_imdb[netflix_id] = imdb_id
                fp = gzip.open('/data/imdb/' + imdb_id + ".p", 'wb')
                cPickle.dump(movie, fp)
                fp.close()
            except Exception as e:
                print e
                print title

    cPickle.dump(netflix_imdb, open("Neflix_IMDB.p", "wb"))


def import_imdb():
    i = IMDb()
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))
    no_year = neflix_title(with_year=False)
    for id, title in neflix_title().items():
        netflix_id = id + 1
        if netflix_id not in netflix_imdb:
            try:
                ms = i.search_movie(title)
                if len(ms) > 0:
                    m = ms[0]
                else:
                    ms = i.search_movie(no_year[id])
                    if len(ms) > 0:
                        m = ms[0]
                    else:
                        short_name = re.split("[:\(/]+|(Season \d)|(Trilogy)", no_year[id])[0]
                        print "%s -> %s" % (no_year[id], short_name)
                        m = i.search_movie(short_name)[0]

                imdb_id = m.movieID

                movie = i.get_movie(m.movieID)
                movie["keywords"] = i.get_movie_keywords(m.movieID)["data"]
                # movie["votes_details"] = i.get_movie_vote_details(imdb_id)["data"]
                # movie["critic_reviews"] = i.get_movie_critic_reviews(imdb_id)["data"]
                # movie["awards"] = i.get_movie_awards(imdb_id)["data"]
                movie["business"] = i.get_movie_business(imdb_id)["data"]
                # movie["recommendations"] = i.get_movie_recommendations(imdb_id)["data"]
                movie["synopsis"] = i.get_movie_synopsis(imdb_id)["data"]
                # movie["news"] = i.get_movie_news(imdb_id)

                netflix_imdb[netflix_id] = imdb_id
                fp = gzip.open('/data/imdb/' + imdb_id + ".p", 'wb')
                cPickle.dump(movie, fp)
                fp.close()
            except Exception as e:
                print e
                print title


def import_imdb_more():
    i = IMDb()
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))
    bar = pyprind.ProgBar(len(netflix_imdb))
    for netflix_id, imdb_id in netflix_imdb.items():


        imdb_id = netflix_imdb[netflix_id]
        movie = cPickle.load(gzip.open('/data/imdb/' + imdb_id + ".p", 'rb'))
        print movie
        bar.update()
        if "recommendations" in movie.keys():
            continue
        while True:
            try:
                movie["votes_details"] = i.get_movie_vote_details(imdb_id)["data"]
                movie["critic_reviews"] = i.get_movie_critic_reviews(imdb_id)["data"]
                movie["awards"] = i.get_movie_awards(imdb_id)["data"]
                movie["business"] = i.get_movie_business(imdb_id)["data"]
                movie["recommendations"] = i.get_movie_recommendations(imdb_id)["data"]
                movie["synopsis"] = i.get_movie_synopsis(imdb_id)["data"]
                movie["news"] = i.get_movie_news(imdb_id)
                f = gzip.open('/data/imdb/' + imdb_id + ".p", 'wb')
                cPickle.dump(movie, f)
                f.close()
                break
            except Exception as e:
                print e
                time.sleep(2)




def neflix_plot(threshold=10):
    from textblob import TextBlob
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))
    bar = pyprind.ProgBar(len(netflix_imdb))
    movie_plot = {}
    for neflix_id, imdb_id in netflix_imdb.items():
        movie = cPickle.load(gzip.open('/data/imdb/' + imdb_id + ".p", 'rb'))
        tokens = []
        if "keywords" in movie["keywords"]:
            keywords = [k for k in movie["keywords"]["keywords"] if "n-title" not in k]
            for k in keywords:
                tokens.append(k)
                for sub_k in k.split("-"):
                    tokens.append(sub_k)
        if movie.get("plot") is not None:
            plot = movie.get("title") + " " + " ".join(movie.get("plot"))
            if movie.get("plot outline") is not None:
                plot = plot + " " + movie.get("plot outline")

            blob = TextBlob(plot)
            raw_tokens = blob.noun_phrases  + blob.words
            for t in raw_tokens:
                tokens.append(t.lemmatize().lower())
        movie_plot[neflix_id-1] = tokens
        bar.update()

    movie_plot_text = []
    for id in range(len(neflix_title())):
        if id in movie_plot:
            movie_plot_text.append(" ".join([word.replace(" ", "-") for word in movie_plot[id]]))
        else:
            movie_plot_text.append("")

    return  movie_plot_text

def neflix_features(threshold=10):
    from textblob import TextBlob
    netflix_imdb = cPickle.load(open("Neflix_IMDB.p"))
    bar = pyprind.ProgBar(len(netflix_imdb))

    people_dict = {}
    keywords_dict = {}
    others_dict = {}
    for neflix_id, imdb_id in netflix_imdb.items():
        movie = cPickle.load(gzip.open('/data/imdb/' + imdb_id + ".p", 'rb'))
        people = set()
        director = movie.get('director')
        writer = movie.get('writer')
        producer = movie.get('producer')
        composer = movie.get('composer')
        if movie.get('cast') is not None:
            cast = movie.get('cast')[0:min(5,len(movie.get('cast')))]
        else:
            cast = None
        for fs in [director, cast]: #writer, producer, composer,
            if fs is not None:
                people.update(fs)
        people_dict[neflix_id - 1] = people

        # keywords
        if "keywords" in movie["keywords"]:
            keywords = [k for k in movie["keywords"]["keywords"] if "n-title" not in k]
        else:
            keywords = []
        keywords_dict[neflix_id - 1] = set(keywords)
        # other stuff
        other = set()
        country = movie.get('country', u'')
        genres = movie.get('genres')
        year = ["Year " + str((int(movie.get('year'))/10) * 10)] if movie.get('year') is not None else []
        for fs in [country, genres, year]:
            if fs is not None:
                other.update(fs)
        others_dict[neflix_id - 1] = other
        bar.update()


    return people_dict, keywords_dict, others_dict



def neflix_features_more(threshold=10):

    return people_dict, keywords_dict, others_dict

def mix_dict(people_dict, keywords_dict, others_dict, people_thres=4, keyword_thres=30):
    import numpy
    n_items = len(neflix_title())
    p, p_label = feature_sets_to_array(people_dict, people_thres, n_items)
    k, k_label = feature_sets_to_array(keywords_dict, keyword_thres, n_items)
    o, o_label = feature_sets_to_array(others_dict, 1, n_items)
    return numpy.concatenate((p,k,o), axis=1), p_label + k_label + o_label

def dict_2_hash(k, dim, n_items):
    from sklearn.feature_extraction import FeatureHasher
    features = []
    for i in range(n_items):
        if i in k:
            features.append(dict.fromkeys([str(f) for f in k[i]], 1))
        else:
            features.append({})
    h = FeatureHasher(dim)
    return h.fit_transform(features)

class Autoencoder:

    def __init__(self,ins, noises, outs, losses, dims, l1, l2):
        from theano.gof import Variable
        import theano.tensor as T
        import theano
        activation_fn = T.nnet.relu
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.bias = []
        self.hidden = []
        self.total_hidden_dimensions = 0

        for input in ins:
            in_dim = input.shape[1]
            # convert data to shared variable if they are not so
            self.inputs.append(
                theano.shared(numpy.asarray(input, dtype="float32")) if not isinstance(input, Variable) else input)
            ret = self.inputs[-1]
            out_dim = max(in_dim, dims)
            self.total_hidden_dimensions += out_dim
            self.weights += [
                theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (in_dim + out_dim)),
                        high=numpy.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True),
            ]
            self.bias += [theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (in_dim + out_dim)),
                    high=numpy.sqrt(6. / (in_dim + out_dim)),
                    size=(out_dim,)
                ),
                dtype=theano.config.floatX
            ), borrow=True)]
            self.hidden.append(activation_fn(T.dot(ret, self.weights[-1]) + self.bias[-1]))
        self.concat_hidden_layer = T.concatenate(self.hidden, axis=1)

        # Concat Hidden Layer To Share Representation (total_hidden_dims -> dims)
        in_dim = self.total_hidden_dimensions
        out_dim = dims
        self.weights += [
            theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (in_dim + out_dim)),
                    high=numpy.sqrt(6. / (in_dim + out_dim)),
                    size=(in_dim, out_dim)
                ),
                dtype=theano.config.floatX
            ), borrow=True),
        ]
        self.bias += [theano.shared(numpy.asarray(
            numpy.random.uniform(
                low=-numpy.sqrt(6. / (in_dim + out_dim)),
                high=numpy.sqrt(6. / (in_dim + out_dim)),
                size=(out_dim,)
            ),
            dtype=theano.config.floatX
        ), borrow=True)]

        # Encoding Encoding
        self.representation = activation_fn(T.dot(self.concat_hidden_layer, self.weights[-1]) + self.bias[-1])

        # Reconstruction
        self.cost = 0.0
        for output, loss in zip(outs, losses):
            out_dim = output.shape[1]
            self.outputs.append(theano.shared(numpy.asarray(output, dtype="float32")) if not isinstance(output, Variable) else output)
            output = self.outputs[-1]
            in_dim = dims
            self.weights += [
                theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (in_dim + out_dim)),
                        high=numpy.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True),
            ]
            self.bias += [theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (in_dim + out_dim)),
                    high=numpy.sqrt(6. / (in_dim + out_dim)),
                    size=(out_dim,)
                ),
                dtype=theano.config.floatX
            ), borrow=True)]
            # Hidden Layer Before Reconstruction
            hidden = activation_fn(T.dot(self.representation, self.weights[-1]) + self.bias[-1])

            # Reconstruction
            in_dim = out_dim
            self.weights += [
                theano.shared(numpy.asarray(
                    numpy.random.uniform(
                        low=-numpy.sqrt(6. / (in_dim + out_dim)),
                        high=numpy.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ),
                    dtype=theano.config.floatX
                ), borrow=True),
            ]
            self.bias += [theano.shared(numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (in_dim + out_dim)),
                    high=numpy.sqrt(6. / (in_dim + out_dim)),
                    size=(out_dim,)
                ),
                dtype=theano.config.floatX
            ), borrow=True)]

            # Output Layer
            reconstruct = T.dot(hidden, self.weights[-1]) + self.bias[-1]
            if loss == "l2":
                self.cost += ((reconstruct-output)**2).sum(axis=1).mean()
            elif loss == "cross_entropy":
                reconstruct = T.nnet.sigmoid(reconstruct)
                self.cost -= T.sum(output * T.log(reconstruct) + (1 - output) * T.log(1 - reconstruct), axis=1).mean()
            else:
                print "No loss"

        self.updates = []

        # Regularization
        self.lr = T.fscalar()
        for variable in self.weights:
            self.cost += (variable ** 2).sum() * l2
            self.cost += T.abs_(variable).sum() * l1

        # Gradient
        for variable in self.bias + self.weights:
            g = T.grad(self.cost, wrt=variable)
            self.updates.append([variable, variable - (g*self.lr)])

        self.f = theano.function(inputs=[self.lr], outputs=self.cost, updates=self.updates)



