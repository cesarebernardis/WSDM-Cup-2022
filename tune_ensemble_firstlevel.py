import os
import optuna
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from xgboost_utils import XGBoostOptimizer, XGBoostSmallOptimizer
from lightgbm_utils import LightGBMOptimizer, LightGBMSmallOptimizer
from catboost_utils import CatboostOptimizer, CatboostSmallOptimizer

from utils import FeatureGenerator, Optimizer, remove_seen, remove_useless_features, get_useless_columns
from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble, stretch_urm, make_submission


EXPERIMENTAL_CONFIG['n_folds'] = 0


def print_importance(model, n=10):
    names = model.feature_name_
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:n]:
        print(names[i], importances[i])




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='First level Ensemble hyperparameter optimization')
    parser.add_argument('--ntrials', '-t', metavar='TRIALS', type=int, nargs='?', default=200,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--nfolds', '-cv', metavar='FOLDS', type=int, nargs='?', default=5,
                        help='Number of CV folds for hyperparameter optimization')
    parser.add_argument('--force-hpo', '-f', nargs='?', dest="force_hpo", default=False, const=True,
                        help='Whether to run a new hyperparameter optimization discarding previous ones')
    parser.add_argument('--small', '-s', nargs='?', dest="small", default=False, const=True,
                        help='Whether to train on two different splittings, but on smaller train sets')
    parser.add_argument('--gbdt', '-g', metavar='GBDT', type=str, nargs='?', default='lightgbm', dest="gbdt",
                        choices=['lightgbm', 'xgboost', 'catboost'], help='GBDT model to use')

    args = parser.parse_args()
    n_trials = args.ntrials
    n_folds = args.nfolds
    force_hpo = args.force_hpo

    if args.gbdt == "xgboost":
        if args.small:
            optimizer_class = XGBoostSmallOptimizer
        else:
            optimizer_class = XGBoostOptimizer
    elif args.gbdt == "catboost":
        if args.small:
            optimizer_class = CatboostSmallOptimizer
        else:
            optimizer_class = CatboostOptimizer
    else:
        if args.small:
            optimizer_class = LightGBMSmallOptimizer
        else:
            optimizer_class = LightGBMOptimizer

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

                useless_cols = get_useless_columns(ratings[-2])
                for i in range(len(ratings)):
                    remove_useless_features(ratings[i], columns_to_remove=useless_cols, inplace=True)

                user_factors, item_factors = featgen.load_user_factors(folder, num_factors=12, normalize=True)
                for j in range(len(user_factors)):
                    ratings[j] = ratings[j].merge(item_factors[j], on=["item"], how="left", sort=False)
                    ratings[j] = ratings[j].merge(user_factors[j], on=["user"], how="left", sort=True)

                optimizer = optimizer_class(urms, ratings, validations, n_folds=n_folds)
                optimizer.optimize_all(exam_folder, force=force_hpo, n_trials=n_trials, folder=folder, study_name_suffix="-f")

                results_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + \
                                   "{}-ensemble-prediction-{}".format(optimizer.NAME, exam_folder)

                er, er_test, result = optimizer.train_cv_best_params(urms[-2], ratings[-2], validations[-1], test_df=ratings[-1])
                output_scores(results_filename + "-valid.tsv.gz", er.tocsr(), user_mappers[-2], item_mappers[-2], compress=True)
                output_scores(results_filename + "-test.tsv.gz", er_test.tocsr(), user_mappers[-1], item_mappers[-1], compress=True)
                print(exam_folder, folder, "Optimization finished:", result)

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
