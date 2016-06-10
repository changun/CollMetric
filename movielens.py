from lightfm.datasets import fetch_movielens
from collections import defaultdict
from utils import *
from model import *
def coo_to_dict(m):
    dat = {}
    rows, cols = m.nonzero()
    for i, u in enumerate(rows):
        if u not in dat:
            dat[u] = set()
        dat[u].add(cols[i])
    return  dat

movielens = fetch_movielens()
movie_train = coo_to_dict(movielens["train"])
movie_test = coo_to_dict(movielens["test"])


movie1 = KBPRModel(50, movielens["train"].shape[0], movielens["train"].shape[1],
                        per_user_sample=50, learning_rate=0.01, lambda_u=1, K=1, margin=1, variance_mu=1, update_mu=False)

early_stopping(movie1, movie_train, movie_test, movie_test, pre= "Movie" + " ",
               valid_per_user_sample=50, start_adagrad=10, start_hard_case=10)
