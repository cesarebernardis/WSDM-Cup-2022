import os
import optuna
import argparse
import lightgbm as lgbm

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

from utils import FeatureGenerator, Optimizer, remove_seen, remove_useless_features, get_useless_columns
from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble, stretch_urm, make_submission


EXPERIMENTAL_CONFIG['n_folds'] = 0


def print_importance(model, n=10):
    names = model.feature_name_
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:n]:
        print(names[i], importances[i])


class LightGBMOptimizer(Optimizer):

    NAME = "lgbm"

    def evaluate(self, predictions_matrix, val, cutoff=10):
        return evaluate(predictions_matrix, val, cutoff=cutoff)

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None):

        model = lgbm.LGBMRanker(**_params)
        splitter = GroupKFold(self.n_folds)
        _ratings = _ratings.sort_values(by=['user'])

        validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
        validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1})
        validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True).fillna(0)

        _r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
        test_r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)

        np.random.seed(13)
        for train_index, test_index in splitter.split(_ratings, validation_df, _ratings.user.values):

            # Ensure to keep order
            train_index = np.sort(train_index)
            test_index = np.sort(test_index)

            users, counts = np.unique(_ratings.user.values[train_index], return_counts=True)
            train_groups = counts[np.argsort(users)]

            users, counts = np.unique(_ratings.user.values[test_index], return_counts=True)
            test_groups = counts[np.argsort(users)]

            train_df = _ratings.drop(['user', 'item'], axis=1)
            pred_df = validation_df.sort_values(by=['user']).drop(['user', 'item'], axis=1)
            model.fit(
                train_df.iloc[train_index], pred_df.iloc[train_index], group=train_groups,
                eval_set=[(train_df.iloc[test_index], pred_df.iloc[test_index])],
                eval_group=[test_groups], eval_at=(10), eval_metric="ndcg",
                callbacks=[lgbm.early_stopping(20, first_metric_only=False, verbose=False), lgbm.log_evaluation(0)]
            )
            predictions = model.predict(train_df.iloc[test_index], raw_score=True) + 1e-10
            predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                               _ratings.item.values[test_index])),
                                    shape=_validation.shape)
            _r += remove_seen(predictions_matrix, _urm)

            if test_df is not None:
                predictions = model.predict(test_df.drop(['user', 'item'], axis=1), raw_score=True) + 1e-10
                predictions_matrix = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                                    shape=_validation.shape)
                predictions_matrix = remove_seen(predictions_matrix, _urm + _validation)
                test_r += predictions_matrix / splitter.get_n_splits()

        user_ndcg, n_evals = self.evaluate(_r, _validation, cutoff=10)
        _r = row_minmax_scaling(_r).tocoo()
        part_score = np.sum(user_ndcg) / n_evals

        if test_df is not None:
            return _r, row_minmax_scaling(test_r).tocoo(), part_score

        return _r, part_score


    def get_params_from_trial(self, trial):

        subsample_freq = trial.suggest_int("subsample_freq", 0, 10)
        if subsample_freq > 0:
            subsample = trial.suggest_float("subsample", 1e-3, 0.9, log=True)
        else:
            subsample = 1.

        return {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 8, 150),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 40, 500),
            "subsample_for_bin": trial.suggest_int("subsample_for_bin", 100, 100000),
            "objective": trial.suggest_categorical("objective", ["lambdarank"]),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-9, 1e-2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10., log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 8, 50),
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "random_state": trial.suggest_categorical("random_state", [1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-2, 1., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 0.1, log=True),
        }




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LightGBM final ensemble hyperparameter optimization')
    parser.add_argument('--ntrials', '-t', metavar='TRIALS', type=int, nargs='?', default=250,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--nfolds', '-cv', metavar='FOLDS', type=int, nargs='?', default=5,
                        help='Number of CV folds for hyperparameter optimization')
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

        for folder in EXPERIMENTAL_CONFIG['datasets']:

            if exam_folder in folder:

                print("Loading", folder)
                ratings = featgen.load_folder_features(folder)

                predictions = featgen.load_algorithms_predictions(folder)
                for j in range(len(predictions)):
                    ratings[j] = ratings[j].merge(predictions[j], on=["user" ,"item"], how="left", sort=True)

                user_factors, item_factors = featgen.load_user_factors(folder, num_factors=12, normalize=True)
                for j in range(len(user_factors)):
                    ratings[j] = ratings[j].merge(item_factors[j], on=["item"], how="left", sort=False)
                    ratings[j] = ratings[j].merge(user_factors[j], on=["user"], how="left", sort=True)

                useless_cols = get_useless_columns(ratings[-2])
                for i in range(len(ratings)):
                    remove_useless_features(ratings[i], columns_to_remove=useless_cols, inplace=True)

                optimizer = LightGBMOptimizer(urms, ratings, validations, n_folds=n_folds)
                optimizer.optimize_all(exam_folder, force=force_hpo, n_trials=n_trials, folder=folder, study_name_suffix="-f")

                results_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + \
                                   "lgbm-ensemble-prediction-{}".format(exam_folder)

                er, er_test, result = optimizer.train_cv_best_params(urms[-2], ratings[-2], validations[-1], test_df=ratings[-1])
                output_scores(results_filename + "-valid.tsv.gz", er.tocsr(), user_mappers[-2], item_mappers[-2], compress=True)
                output_scores(results_filename + "-test.tsv.gz", er_test.tocsr(), user_mappers[-1], item_mappers[-1], compress=True)
                print(exam_folder, folder, "LGBM optimization finished:", result)

                for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                    er, result = optimizer.train_cv_best_params(urms[fold], ratings[fold], validations[fold])
                    output_scores(results_filename + "-f{}.tsv.gz".format(fold), er.tocsr(),
                                  user_mappers[fold], item_mappers[fold], compress=True)

                del optimizer

        del validations
        del user_mappers
        del item_mappers
        del urms
        del featgen
