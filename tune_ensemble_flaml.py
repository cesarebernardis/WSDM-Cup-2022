import os
import optuna
import lightgbm as lgbm
import flaml

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



def train_cv_flaml(_urm, _ratings, _validation, log_file_name, test_df=None, n_folds=5, time_budget=600):

    _ratings = _ratings.sort_values(by=['user'])

    validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
    validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1})
    validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True).fillna(0)

    users, counts = np.unique(_ratings.user.values, return_counts=True)
    groups = counts[np.argsort(users)]

    train_df = _ratings.drop(['user', 'item'], axis=1)
    pred_df = validation_df.sort_values(by=['user']).relevance

    #log_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + "flaml_logfile.log"
    log_filename = ""

    model = flaml.AutoML()
    model.fit(
        train_df, pred_df, groups=groups, estimator_list="auto",
        eval_method="cv", n_splits=n_folds, log_file_name=log_filename, ensemble=True, verbose=2,
        task='rank', time_budget=time_budget, retrain_full=True, seed=1
    )

    predictions = model.predict(train_df)
    predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                       _ratings.item.values[test_index])),
                            shape=_validation.shape)
    predictions_matrix = remove_seen(predictions_matrix, _urm)

    user_ndcg, n_evals = evaluate(predictions_matrix, _validation, cutoff=10)
    part_score = np.sum(user_ndcg) / n_evals

    if test_df is not None:
        predictions = model.predict(test_df.drop(['user', 'item'], axis=1))
        test_r = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                            shape=_validation.shape)
        test_r = remove_seen(test_r, _urm + _validation)

    predictions_matrix = row_minmax_scaling(predictions_matrix).tocoo()

    if len(test_r.data) > 0:
        return predictions_matrix, row_minmax_scaling(test_r).tocoo(), part_score

    return predictions_matrix, part_score



if __name__ == "__main__":

    time_budget = 60 * 60

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()

        featgen = FeatureGenerator(exam_folder)
        urms = featgen.get_urms()
        validations = featgen.get_validations()
        user_mappers = featgen.get_user_mappers()
        item_mappers = featgen.get_item_mappers()

        for folder in EXPERIMENTAL_CONFIG['datasets'][:1]:
            if exam_folder in folder:
                print("Loading", folder)
                featgen.load_folder_features(folder, first_level_ensemble_features=False)
                featgen.load_lgbm_ensemble_feature(folder)
                featgen.load_xgb_ensemble_feature(folder)
                featgen.load_ratings_ensemble_feature(folder)

        basic_dfs = featgen.get_final_df()
        print(basic_dfs[-2].head())
        ratings_folder = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        print("Start training")
        er, er_test, result = train_cv_flaml(urms[-2], basic_dfs[-2], validations[-1],
                ratings_folder + "{}-flaml.log".format(exam_folder), test_df=basic_dfs[-1], n_folds=5, time_budget=time_budget)
        print("FINAL ENSEMBLE {}: {:.8f}".format(exam_folder, result))
        output_scores(ratings_folder + "valid_scores.tsv", er.tocsr(), user_mappers[-2], item_mappers[-2],
                      user_prefix=exam_folder, compress=False)
        output_scores(ratings_folder + "test_scores.tsv", er_test.tocsr(), user_mappers[-1], item_mappers[-1],
                      user_prefix=exam_folder, compress=False)

        del basic_dfs
        del validations
        del user_mappers
        del item_mappers
        del urms
        del featgen

    make_submission()