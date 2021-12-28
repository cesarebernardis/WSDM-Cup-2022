import os
import optuna
import catboost
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl

from sklearn.model_selection import GroupKFold

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from utils import FeatureGenerator
from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble, stretch_urm, make_submission


VAL_WEIGHT = 2.
EXPERIMENTAL_CONFIG['n_folds'] = 0


def print_importance(model, n=10):
    names = model.feature_name_
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:n]:
        print(names[i], importances[i])


def remove_seen(predictions, seen):
    seen = seen.tocsr()
    predictions = predictions.tocsr()
    for u in range(seen.shape[0]):
        if predictions.indptr[u] != predictions.indptr[u+1] and seen.indptr[u] != seen.indptr[u+1]:
            pred_indices = predictions.indices[predictions.indptr[u]:predictions.indptr[u+1]]
            seen_indices = seen.indices[seen.indptr[u]:seen.indptr[u+1]]
            min_data = min(predictions.data[predictions.indptr[u]:predictions.indptr[u+1]])
            mask = np.isin(pred_indices, seen_indices, assume_unique=True)
            if sum(mask) > 0:
                predictions.data[np.arange(len(mask))[mask] + predictions.indptr[u]] = min_data - abs(min_data) * 0.02
    return predictions



def train_cv_catboost(_params, _urm, _ratings, _validation, test_df=None, n_folds=5):

    #_params["thread_count"] = 14

    model = catboost.CatBoostRanker(**_params)
    splitter = GroupKFold(n_folds)
    _ratings = _ratings.sort_values(by=['user'])

    validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
    validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1})
    validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True).fillna(0)

    _r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
    test_r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
    part_score = 0.
    np.random.seed(13)
    for train_index, test_index in splitter.split(_ratings, validation_df, _ratings.user.values):

        # Ensure to keep order
        train_index = np.sort(train_index)
        test_index = np.sort(test_index)

        users, counts = np.unique(_ratings.user.values[train_index], return_counts=True)
        train_groups = np.repeat(np.sort(users), counts[np.argsort(users)])

        users, counts = np.unique(_ratings.user.values[test_index], return_counts=True)
        validation_groups = np.repeat(np.sort(users), counts[np.argsort(users)])

        train_df = _ratings.drop(['user', 'item'], axis=1)
        pred_df = validation_df.sort_values(by=['user']).drop(['user', 'item'], axis=1)

        train = catboost.Pool(
            data=train_df.iloc[train_index],
            label=pred_df.iloc[train_index],
            group_id=train_groups
        )

        validation = catboost.Pool(
            data=train_df.iloc[test_index],
            label=pred_df.iloc[test_index],
            group_id=validation_groups
        )

        model.fit(train, eval_set=validation, verbose_eval=False, early_stopping_rounds=10)
        predictions = model.predict(validation)
        predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                           _ratings.item.values[test_index])),
                                shape=_validation.shape)
        predictions_matrix = remove_seen(predictions_matrix, _urm)

        weights = np.zeros(_validation.shape[0], dtype=np.float32)
        weights[np.unique(_ratings.user.values[test_index])] = 1.
        val = sps.diags(weights, dtype=np.float32).dot(_validation)
        val.eliminate_zeros()

        user_ndcg, n_evals = evaluate(predictions_matrix, val, cutoff=10)
        part_score += np.sum(user_ndcg) / n_evals
        _r += predictions_matrix

        if test_df is not None:
            users, counts = np.unique(test_df.user.values, return_counts=True)
            test_groups = np.repeat(np.sort(users), counts[np.argsort(users)])
            test_df = test_df.sort_values(by=['user'])
            test = catboost.Pool(
                data=test_df.drop(['user', 'item'], axis=1),
                group_id=test_groups
            )
            predictions = model.predict(test)
            predictions_matrix = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                                shape=_validation.shape)
            predictions_matrix = remove_seen(predictions_matrix, _urm + _validation)
            test_r += predictions_matrix / splitter.get_n_splits()

    _r = row_minmax_scaling(_r).tocoo()
    part_score /= splitter.get_n_splits()

    if len(test_r.data) > 0:
        return _r, row_minmax_scaling(test_r).tocoo(), part_score

    return _r, part_score




def objective(trial, _urms, _ratings, _validations):

    _val_weight = VAL_WEIGHT
    final_score = 0.
    denominator = 0.
    bagging_temperature = None
    subsample = None
    
    grow_policy = None #trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])
    objective = trial.suggest_categorical("objective", ["YetiRank", "YetiRankPairwise", 
        #"StochasticFilter", "StochasticRank:metric=NDCG;top=10", "StochasticRank:metric=DCG;top=10"
        "QuerySoftMax", "QueryRMSE"
        ])

    sampling_unit = None
    if objective == "YetiRankPairwise":
        sampling_unit = trial.suggest_categorical("sampling_unit", ["Object", "Group"])

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"])

    if bootstrap_type == "Bayesian":
        bagging_temperature = trial.suggest_float("bagging_temperature", 1e-2, 100., log=True)
    else:
        subsample = trial.suggest_float("subsample",  1e-2, 0.7, log=True)
    
    max_max_depth = 12
    if objective in ["YetiRank", "YetiRankPairwise"]:
        max_max_depth = 8

    params = {
        "objective": objective,
        "custom_metric": trial.suggest_categorical("custom_metric", ["NDCG:top=10"]),
        "eval_metric": trial.suggest_categorical("eval_metric", ["NDCG:top=10"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.4, log=True),
        "grow_policy": grow_policy,
        "iterations": trial.suggest_categorical("iterations", [1500]),
        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
        "max_depth": trial.suggest_int("max_depth", 2, max_max_depth),
        "random_state": trial.suggest_categorical("random_state", [1]),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10., log=True),
        "bootstrap_type": bootstrap_type,
        "subsample": subsample,
        "bagging_temperature": bagging_temperature,
        #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1., log=True),
        "score_function": trial.suggest_categorical("score_function", ["Cosine", "L2"]),
        "sampling_unit": sampling_unit,
    }

    for fold in range(len(_validations)):
        _, part_score = train_cv_catboost(params, _urms[fold], _ratings[fold], _validations[fold])
        final_score += part_score * _val_weight
        denominator += _val_weight
        _val_weight = 1.

    return final_score / denominator


def optimize(study_name, storage, urms, ratings, validations, force=False, n_trials=250):
    sampler = optuna.samplers.TPESampler(n_startup_trials=50)
    if force:
        try:
            optuna.study.delete_study(study_name, storage)
        except KeyError:
            pass
    _objective = lambda x: objective(x, urms, ratings, validations)
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler,
                                load_if_exists=True, storage=storage)
    if n_trials - len(study.trials) > 0:
        study.optimize(_objective, n_trials=n_trials - len(study.trials), show_progress_bar=True)
    return study


def optimize_all(exam_folder, urms, ratings, validations, force=False, n_trials=250, folder=None):
    result_dict = {}
    storage = "sqlite:///" + EXPERIMENTAL_CONFIG['dataset_folder']
    if folder is not None:
        storage += folder + os.sep + "optuna.db"
    else:
        storage += exam_folder + os.sep + "optuna.db"
    if folder is not None:
        study_name = "catboost-ensemble-{}-{}".format(folder, exam_folder)
    else:
        study_name = "catboost-last-level-ensemble-{}".format(exam_folder)
    study = optimize(study_name, storage, urms, ratings, validations, force=force, n_trials=n_trials)
    return study.best_params



if __name__ == "__main__":

    n_trials = 250

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()

        featgen = FeatureGenerator(exam_folder)

        urms = featgen.get_urms()
        validations = featgen.get_validations()
        user_mappers = featgen.get_user_mappers()
        item_mappers = featgen.get_item_mappers()

        for folder in EXPERIMENTAL_CONFIG['datasets']:

            if exam_folder in folder:

                ratings = featgen.load_folder_features(folder, add_dataset_features_to_last_level_df=False)

                predictions = featgen.load_algorithms_predictions(folder)
                for j in range(len(predictions)):
                    ratings[j] = ratings[j].merge(predictions[j], on=["user", "item"], how="left", sort=True)

                user_factors, item_factors = featgen.load_user_factors(folder, num_factors=12, normalize=True)
                for j in range(len(user_factors)):
                    ratings[j] = ratings[j].merge(item_factors[j], on=["item"], how="left", sort=False)
                    ratings[j] = ratings[j].merge(user_factors[j], on=["user"], how="left", sort=True)

                result_dict = optimize_all(exam_folder, urms, ratings, validations,
                                           force=False, n_trials=n_trials, folder=folder)

                results_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + \
                                   "catboost-ensemble-prediction-{}".format(exam_folder)

                er, er_test, result = train_cv_catboost(
                    result_dict, urms[-2], ratings[-2], validations[-1], test_df=ratings[-1], n_folds=5)
                output_scores(results_filename + "-valid.tsv.gz", er.tocsr(), user_mappers[-2], item_mappers[-2],
                              user_prefix=exam_folder, compress=True)
                output_scores(results_filename + "-test.tsv.gz", er_test.tocsr(), user_mappers[-1], item_mappers[-1],
                              user_prefix=exam_folder, compress=True)
                print(exam_folder, folder, "CatBoost optimization finished:", result)

                for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                    er, result = train_cv_catboost(result_dict, urms[fold], ratings[fold], validations[fold], n_folds=5)
                    output_scores(results_filename + "-f{}.tsv.gz".format(fold), er.tocsr(), user_mappers[fold], item_mappers[fold],
                              user_prefix=exam_folder, compress=True)

        del basic_dfs
        del validations
        del user_mappers
        del item_mappers
        del urms
        del featgen

    make_submission()
