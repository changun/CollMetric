from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)


def objective(param):
    from model import KBPRModel
    from utils import early_stopping, bookcx, preprocess
    import gc
    n_items_bc, n_users_bc, train_dict_bc, valid_dict_bc, test_dict_bc, exclude_dict_bc = \
        preprocess(bookcx(), portion=[8, 1, 1, 0])
    bc_50 = KBPRModel(50, n_users_bc, n_items_bc, batch_size=-1,
                      per_user_sample=1000, learning_rate=0.1, lambda_u=0.0,
                      lambda_v=0.0, use_bias=True,
                      K=1, margin=1, variance_mu=1, update_mu=True, lambda_variance=1,
                      normalization=False, uneven_sample=True,
                      use_warp=True, **param)
    best_metric = early_stopping(bc_50, train_dict_bc, valid_dict_bc, test_dict_bc,
                                 lambda m: -m.recall(valid_dict_bc, train_dict_bc)[0], save_model=False,
                                 valid_per_user_sample=100, start_hard_case=10000, start_adagrad=0000, n_epochs=10000,
                                 patience=1000)
    del bc_50
    gc.collect()
    return best_metric


trials = MongoTrials('mongo://localhost/exp_bookcx_bpr_50/jobs', exp_key='BookCX_50')
best = fmin(objective,
            space={"lambda_bias": hp.loguniform('lambda_bias', -5, 1),
                   "lambda_cov": hp.loguniform('lambda_cov', -5, 5),
                   "max_norm": hp.uniform('max_norm', 1, 2),
                   },
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

trials_bias = MongoTrials('mongo://localhost/exp_bookcx_bpr_50_bias/jobs', exp_key='BookCX_50_Bias')
best_bias = fmin(objective,
                 space={"lambda_bias": hp.loguniform('lambda_bias', -5, 0),
                        "lambda_cov": hp.choice("lambda_cov", [50.0]),
                        "max_norm": hp.choice("max_norm", [1.3])
                        },
                 algo=tpe.suggest,
                 max_evals=300,
                 trials=trials)

from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)


def objective_lightfm(param):
    from model import LightFMModel
    from utils import bookcx, preprocess
    import numpy
    import gc
    lambda_uv = param["lambda_uv"]
    n_items_bc, n_users_bc, train_dict_bc, valid_dict_bc, test_dict_bc, exclude_dict_bc = \
        preprocess(bookcx(), portion=[8, 1, 1, 0])
    bc_warp = LightFMModel(50, n_users_bc, n_items_bc, lambda_v=lambda_uv, lambda_u=lambda_uv)
    print bc_warp
    best_metric = numpy.Infinity
    patience = 6
    for i in range(100):
        bc_warp.train(train_dict_bc, epoch=50)
        new_metric = -bc_warp.recall(valid_dict_bc, train_dict_bc, n=100)[0]
        if new_metric > best_metric:
            patience -= 1
            print ("Patience %d" % (patience,))
        else:
            patience = 6
            best_metric = new_metric
        if patience == 0:
            break
    del bc_warp
    gc.collect()
    return best_metric


warp_trials = MongoTrials('mongo://localhost/exp_bookcx_warp_50/jobs', exp_key='BookCX_50_WARP')
best_warp = fmin(objective_lightfm,
                 space={"lambda_uv": hp.loguniform('lambda_uv', -10, -2)},
                 algo=tpe.suggest,
                 max_evals=100,
                 trials=warp_trials)

## LightFM
from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)


def objective_lightfm_5(param):
    from model import LightFMModel
    from utils import bookcx, preprocess
    import numpy
    import gc
    lambda_uv = param["lambda_uv"]
    n_items_bc, n_users_bc, train_dict_bc, valid_dict_bc, test_dict_bc, exclude_dict_bc = \
        preprocess(bookcx(item_thres=5), portion=[8, 1, 1, 0])
    bc_warp = LightFMModel(50, n_users_bc, n_items_bc, lambda_v=lambda_uv, lambda_u=lambda_uv)
    print bc_warp

    del bc_warp
    gc.collect()
    return best_metric


warp_trials_5 = MongoTrials('mongo://localhost/exp_bookcx_warp_50_5/jobs', exp_key='BookCX_50_WARP_5')
best_warp_5 = fmin(objective_lightfm_5,
                   space={"lambda_uv": hp.loguniform('lambda_uv', -10, -4)},
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=warp_trials_5)

## Ours
from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)

from model import KBPRModel, MaxMF
from utils import early_stopping, bookcx, preprocess


def objective_50_5(param):
    from model import KBPRModel
    from utils import early_stopping, bookcx, preprocess
    import gc
    n_items_bc, n_users_bc, train_dict_bc, valid_dict_bc, test_dict_bc, exclude_dict_bc = \
        preprocess(bookcx(item_thres=5), portion=[8, 1, 1, 0])
    bc_50 = KBPRModel(50, n_users_bc, n_items_bc, batch_size=-1,
                      per_user_sample=50, learning_rate=0.1, lambda_u=0.0,
                      lambda_v=0.0, use_bias=True,
                      K=1, margin=1, variance_mu=1, update_mu=True, lambda_variance=1,
                      normalization=False, uneven_sample=True,
                      use_warp=True, **param)
    best_metric = early_stopping(bc_50, train_dict_bc, valid_dict_bc, test_dict_bc,
                                 lambda m: -m.recall(valid_dict_bc, train_dict_bc)[0], save_model=False,
                                 valid_per_user_sample=100, start_hard_case=10000, start_adagrad=0000, n_epochs=10000,
                                 patience=1000, validation_frequency=100)
    del bc_50
    gc.collect()
    return best_metric


trials_50_5 = MongoTrials('mongo://localhost/exp_bookcx_bpr_50_5/jobs', exp_key='BookCX_50')
best_50_5 = fmin(objective_50_5,
                 space={"lambda_bias": hp.loguniform('lambda_bias', -1, 1),
                        "lambda_cov": hp.loguniform('lambda_cov', 3, 5),
                        "max_norm": hp.uniform('max_norm', 1.1, 1.4),
                        "uneven_sample": hp.choice("uneven_sample", [True, False])
                        },
                 algo=tpe.suggest,
                 max_evals=300,
                 trials=trials_50_5)

n_items_bc, n_users_bc, train_dict_bc, valid_dict_bc, test_dict_bc, exclude_dict_bc = \
    preprocess(bookcx(item_thres=5), portion=[8, 1, 1, 0])
bc_50 = KBPRModel(50, n_users_bc, n_items_bc, batch_size=-1,
                  per_user_sample=100, learning_rate=0.1, lambda_u=0.0,
                  lambda_v=0.0, use_bias=True,
                  K=1, margin=1, variance_mu=1, update_mu=True, lambda_variance=1,
                  normalization=False, uneven_sample=False,
                  use_warp=True, lambda_cov=10, max_norm=1.3, lambda_bias=1)
best_metric = early_stopping(bc_50, train_dict_bc, valid_dict_bc, test_dict_bc,
                             lambda m: -m.recall(valid_dict_bc, train_dict_bc)[0], save_model=False,
                             valid_per_user_sample=100, start_hard_case=10000, start_adagrad=0000, n_epochs=10000,
                             patience=1000)

## MovieLens10M

from model import LightFMModel
from utils import movielens10M, preprocess, early_stop_for_lightfm

n_items_10m, n_users_10m, train_dict_10m, valid_dict_10m, test_dict_10m, exclude_dict_10m = \
    preprocess(movielens10M(4.0), portion=[8, 1, 1, 0])
bc_warp = LightFMModel(50, n_users_10m, n_items_10m, lambda_v=0.0001, lambda_u=0.0001)
best_metric = early_stop_for_lightfm(bc_warp, train_dict_10m, lambda m: -m.recall(valid_dict_10m, train_dict_10m)[0])

## Ours

from model import KBPRModel
from utils import movielens10M, preprocess, early_stopping

n_items_10m, n_users_10m, train_dict_10m, valid_dict_10m, test_dict_10m, exclude_dict_10m = \
    preprocess(movielens10M(4.0), portion=[8, 1, 1, 0])
movie_10m_50 = KBPRModel(50, n_users_10m, n_items_10m, batch_size=-1,
                         per_user_sample=10, learning_rate=0.1, lambda_u=0.0,
                         lambda_v=0.0, use_bias=True,
                         K=1, margin=1, variance_mu=1, update_mu=True, lambda_variance=1,
                         normalization=False, uneven_sample=False,
                         use_warp=True, lambda_cov=10, max_norm=1.3, lambda_bias=1)
best_metric = early_stopping(movie_10m_50, train_dict_10m, valid_dict_10m, test_dict_10m,
                             lambda m: -m.recall(valid_dict_10m, train_dict_10m, n_users=1000)[0], save_model=False,
                             valid_per_user_sample=10, start_hard_case=10000, start_adagrad=0000, n_epochs=10000,
                             patience=1000, validation_frequency=20)
## Flickr

from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)


def flickr_objective_50(param):
    from model import KBPRModel
    from utils import early_stop, flickr, preprocess
    import gc, theano.misc.pkl_utils
    from hyperopt import STATUS_OK
    import cStringIO
    n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict = \
        preprocess(flickr(), portion=[8, 1, 1, 0])
    flickr_50 = KBPRModel(50, n_users, n_items,
                          per_user_sample=50,
                          learning_rate=0.1,
                          variance_mu=1,
                          update_mu=True,
                          lambda_variance=1,
                          use_warp=True, **param)
    best_metric, best_model = early_stop(flickr_50, train_dict,
                                         lambda m: -m.recall(valid_dict, train_dict, n_users=3000)[0],
                                         n_epochs=10000,
                                         patience=500, validation_frequency=100)
    output = cStringIO.StringIO()
    theano.misc.pkl_utils.dump(best_model, output)
    del flickr_50
    gc.collect()

    return {"loss": best_metric, "attachments": {"model": output.getvalue()}, "status": STATUS_OK}


trials_50_flickr = MongoTrials('mongo://localhost/exp_flickr_bpr_50/jobs', exp_key='Flickr_50')
best_50_5 = fmin(flickr_objective_50,
                 space={"lambda_bias": hp.loguniform('lambda_bias', 0, 1),
                        "lambda_cov": hp.loguniform('lambda_cov', 3, 5),
                        "max_norm": hp.uniform('max_norm', 1.1, 1.3),
                        "uneven_sample": hp.choice("uneven_sample", [False])
                        },
                 algo=tpe.suggest,
                 max_evals=300,
                 trials=trials_50_flickr)

# U_3
from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)

