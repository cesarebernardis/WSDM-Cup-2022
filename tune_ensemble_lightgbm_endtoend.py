import os
import argparse
import optuna
import lightgbm as lgbm

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl
import zipfile
import shutil

from eli5.permutation_importance import get_score_importances

from sklearn.model_selection import GroupKFold

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from tune_ensemble_lightgbm import LightGBMOptimizer
from utils import FeatureGenerator, remove_seen, remove_useless_features, get_useless_columns
from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble, stretch_urm, make_submission


VAL_WEIGHT = 2.
EXPERIMENTAL_CONFIG['n_folds'] = 0


def fill_solution(predictions, base_pred, only_missing=False):
    predictions = row_minmax_scaling(predictions).tocoo()
    df1 = pd.DataFrame({'user': predictions.row, 'item': predictions.col, 'final_pred': predictions.data + 2.})
    base_pred = row_minmax_scaling(base_pred).tocoo()
    df2 = pd.DataFrame({'user': base_pred.row, 'item': base_pred.col, 'base_pred': base_pred.data})
    if only_missing:
        in_pred = np.ediff1d(predictions.tocsr().indptr)
        in_base_pred = np.ediff1d(base_pred.tocsr().indptr)
        mask = np.logical_and(in_pred == 0, in_base_pred > 0)
        in_base_pred_matrix = sps.diags(mask, dtype=np.float32).dot(base_pred).tocoo()
        dfp = pd.DataFrame({'user': in_base_pred_matrix.row, 'item': in_base_pred_matrix.col, 'base_pred_present': in_base_pred_matrix.data})
        df2 = df2.merge(dfp, on=["user", "item"], how="left", sort=True)
        df2.base_pred_present.fillna(-0.1, inplace=True)
        df2.base_pred = df2.base_pred_present
    df = df2.merge(df1, on=["user", "item"], how="left", sort=True)
    df.final_pred.fillna(df.base_pred, inplace=True)
    return sps.csr_matrix((df.final_pred, (df.user, df.item)), shape=predictions.shape, dtype=np.float32)


def filter_negatives(users, items, scores, urm_negative, validation=None, tokeep=25, keep_positive_only=False):
    predictions = sps.csr_matrix((scores, (users, items)), shape=urm_negative.shape, dtype=np.float32)
    predictions = row_minmax_scaling(predictions)
    if validation is not None and not keep_positive_only:
        v = validation.tocsr(copy=True)
        v.data[:] = 10.
        predictions += v
        del v
    predictions = predictions.tocsr()
    for u in range(urm_negative.shape[0]):
        start = predictions.indptr[u]
        end = predictions.indptr[u+1]
        if urm_negative.indptr[u] != urm_negative.indptr[u+1]:
            ranking = np.argsort(predictions.data[start:end])[:-tokeep]
            if keep_positive_only and np.all(
                    np.isin(validation.indices[validation.indptr[u]:validation.indptr[u+1]],
                            predictions.indices[ranking + start])
            ):
                predictions.data[start:end] = 0.
            else:
                predictions.data[ranking + start] = 0.
        else:
            predictions.data[start:end] = 0.
    predictions.eliminate_zeros()
    predictions = predictions.tocoo()
    return predictions.row, predictions.col



def load_submission(filename, exam_folder, user_mapping, item_mapping):

    archive = zipfile.ZipFile(filename, 'r')
    extracted_folder = "submission" + os.sep + "tmp" + os.sep

    extracted_filename = extracted_folder + exam_folder + os.sep + 'valid_pred.tsv'
    archive.extract(exam_folder + '/valid_pred.tsv', path=extracted_folder)
    validation =  read_ratings(extracted_filename, user_mapping, item_mapping)

    archive.extract(exam_folder + '/test_pred.tsv', path=extracted_folder)
    extracted_filename = extracted_folder + exam_folder + os.sep + 'test_pred.tsv'
    test =  read_ratings(extracted_filename, user_mapping, item_mapping)

    shutil.rmtree(extracted_folder)
    return validation.tocoo(), test.tocoo()


def print_importance(model, n=10):
    names = model.feature_name_
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:n]:
        print(names[i], importances[i])



class LightGBMTopkOptimizer(LightGBMOptimizer):

    def __init__(self, urms, ratings, validations, reference_baseline, n_folds=5):
        super(LightGBMTopkOptimizer, self).__init__(urms, ratings, validations, n_folds=n_folds)
        self.reference_baseline = reference_baseline

    def evaluate(self, predictions_matrix, val, cutoff=10):
        return evaluate(fill_solution(predictions_matrix, self.reference_baseline, only_missing=True), val, cutoff=cutoff)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LightGBM final ensemble hyperparameter optimization')
    parser.add_argument('--ntrials', '-t', metavar='TRIALS', type=int, nargs='?', default=250,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--nfolds', '-cv', metavar='FOLDS', type=int, nargs='?', default=5,
                        help='Number of CV folds for hyperparameter optimization')
    parser.add_argument('--include-subs', '-i', nargs='?', dest="include_subs", default=False, const=True,
                        help='Whether to include good submissions in the ensemble')
    parser.add_argument('--force-hpo', '-f', nargs='?', dest="force_hpo", default=False, const=True,
                        help='Whether to run a new hyperparameter optimization discarding previous ones')

    args = parser.parse_args()
    n_trials = args.ntrials
    n_folds = args.nfolds
    force_hpo = args.force_hpo

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()

        featgen = FeatureGenerator(exam_folder)

        urms = featgen.get_urms()
        validations = featgen.get_validations()
        user_mappers = featgen.get_user_mappers()
        item_mappers = featgen.get_item_mappers()
        all_predictions = []

        for folder in EXPERIMENTAL_CONFIG['datasets']:
            if exam_folder in folder:
                print("Loading", folder)
                all_predictions.append(featgen.load_algorithms_predictions(folder, only_best_baselines=False))
                all_predictions.append(featgen.load_folder_features(folder, include_fold_features=False))
                featgen.load_ratings_ensemble_feature(folder)
                featgen.load_lgbm_ensemble_feature(folder)
                #featgen.load_catboost_ensemble_feature(folder)
                featgen.load_xgb_ensemble_feature(folder)

        basic_dfs = featgen.get_final_df()
        user_factors, item_factors = featgen.load_user_factors(exam_folder, num_factors=32, epochs=30, reg=5e-5,
                                                               normalize=True, use_val_factors_for_test=False)
        for j in range(len(user_factors)):
            basic_dfs[j] = basic_dfs[j].merge(item_factors[j], on=["item"], how="left", sort=False)
            basic_dfs[j] = basic_dfs[j].merge(user_factors[j], on=["user"], how="left", sort=True)

        for predictions in all_predictions:
            for j in range(len(predictions)):
                basic_dfs[j] = basic_dfs[j].merge(predictions[j], on=["user", "item"], how="left", sort=True)

        if args.include_subs:
            for sub in range(18, 25):
                submission_name = "submission-{}".format(sub)
                filename = "submission" + os.sep + "{}.zip".format(submission_name)
                v, t = load_submission(filename, exam_folder, exam_user_mapper, exam_item_mapper)
                basic_dfs[-2] = basic_dfs[-2].merge(pd.DataFrame({'user': v.row, 'item': v.col, submission_name: v.data}),
                                                    on=["user", "item"], how="left", sort=True)
                basic_dfs[-1] = basic_dfs[-1].merge(pd.DataFrame({'user': t.row, 'item': t.col, submission_name: t.data}),
                                                    on=["user", "item"], how="left", sort=True)

            useless_cols = get_useless_columns(basic_dfs[-2])
            for i in range(len(basic_dfs)):
                remove_useless_features(basic_dfs[i], columns_to_remove=useless_cols, inplace=True)

        ratings_folder = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        optimizer = LightGBMOptimizer(urms, basic_dfs, validations, n_folds=n_folds)
        optimizer.optimize_all(exam_folder, force=force_hpo, n_trials=n_trials, folder=None, study_name_suffix="_endtoend")
        er, er_test, result = optimizer.train_cv_best_params(urms[-2], basic_dfs[-2], validations[-1], test_df=basic_dfs[-1])
        print("FINAL ENSEMBLE {}: {:.8f}".format(exam_folder, result))

        # def score_func(X, y):
        #     _, result = optimizer.train_cv_best_params(urms[-2], X, y)
        #     return result
        #
        # base_score, score_decreases = get_score_importances(score_func, basic_dfs[-2], validations[-1], n_iter=7)
        # feature_importances = np.mean(score_decreases, axis=0)
        # with open("fi-{}.txt".format(exam_folder), "w") as file:
        #     for v in feature_importances:
        #         print(v)

        output_scores(ratings_folder + "valid_scores.tsv", er, user_mappers[-2], item_mappers[-2], compress=False)
        output_scores(ratings_folder + "test_scores.tsv", er_test, user_mappers[-1], item_mappers[-1], compress=False)

        scores, n_evals = evaluate(read_ratings(ratings_folder + "valid_scores.tsv", exam_user_mapper, exam_item_mapper), validations[-1])
        print("FINAL ENSEMBLE {}: {:.8f}".format(exam_folder, np.sum(scores) / n_evals))

        del basic_dfs
        del validations
        del user_mappers
        del item_mappers
        del urms
        del featgen
        del optimizer

    make_submission()