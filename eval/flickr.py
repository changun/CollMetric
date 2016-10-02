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




## LightFM
from hyperopt.mongoexp import MongoTrials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging, sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO)


def objective_lightfm_flickr_50(param):
    from model import LightFMModel
    from utils import flickr, preprocess, early_stop
    import gc, theano.misc.pkl_utils
    import cStringIO
    from hyperopt import STATUS_OK
    lambda_uv = param["lambda_uv"]
    n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict = \
        preprocess(flickr(), portion=[8, 1, 1, 0])
    flickr_warp = LightFMModel(50, n_users, n_items, lambda_v=lambda_uv, lambda_u=lambda_uv)
    best_metric, best_model = early_stop(flickr_warp, train_dict,
                                         lambda m: -m.recall(valid_dict, train_dict, n_users=2000)[0],
                                         n_epochs=10000,
                                         patience=500, validation_frequency=100)

    output = cStringIO.StringIO()
    theano.misc.pkl_utils.dump(best_model, output)
    del flickr_warp
    gc.collect()

    return {"loss": best_metric, "attachments": {"model": output.getvalue()}, "status": STATUS_OK}


warp_trials = MongoTrials('mongo://localhost/exp_flickr_warp_50/jobs', exp_key='Flickr_50_WARP_5')
best_warp_5 = fmin(objective_lightfm_flickr_50,
                   space={"lambda_uv": hp.loguniform('lambda_uv', -10, -4)},
                   algo=tpe.suggest,
                   max_evals=100,
                   trials=warp_trials)



def flickr_objective_50_3(param):
    from model import KBPRModel
    from utils import early_stop, flickr, preprocess
    import gc, theano.misc.pkl_utils
    from hyperopt import STATUS_OK
    import cStringIO
    n_items, n_users, train_dict, valid_dict, test_dict, exclude_dict = \
        preprocess(flickr(), portion=[8, 1, 1, 0])
    flickr_50 = theano.misc.pkl_utils.load(open("flickr_50.p", "rb"))
    flickr_3 = KBPRModel(50, n_users, n_items,
                         U=flickr_50.kmean(train_dict, 3, normalize=False, learning_rate=0.1),
                         V=flickr_50.V.get_value(), b=flickr_50.b.get_value(),
                         per_user_sample=20,
                         learning_rate=0.01,
                         update_mu=True, K=3,
                         lambda_density=0.1,
                         lambda_bias=1.0,
                         uneven_sample=False,
                         max_norm=flickr_50.max_norm,
                         lambda_cov=flickr_50.lambda_cov,
                         lambda_mean_distance=0.0,
                         use_warp=True, **param)
    del flickr_50
    best_metric, best_model = early_stop(flickr_3, train_dict,
                                         lambda m: -m.recall(valid_dict, train_dict, n_users=3000)[0],
                                         n_epochs=10000,
                                         patience=500, validation_frequency=50)
    output = cStringIO.StringIO()
    theano.misc.pkl_utils.dump(best_model, output)
    del flickr_3
    gc.collect()

    return {"loss": best_metric, "attachments": {"model": output.getvalue()}, "status": STATUS_OK}


trials_50_flickr_3 = MongoTrials('mongo://localhost/exp_flickr_bpr_50_3_0/jobs', exp_key='Flickr_50_3')
best_50_3 = fmin(flickr_objective_50_3,
                 space={"lambda_variance": hp.uniform('lambda_variance', 0.05, 1),
                        "variance_mu": hp.uniform('variance_mu', 0.1, 1)
                        },
                 algo=tpe.suggest,
                 max_evals=300,
                 trials=trials_50_flickr_3)
