import glob
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

from lightgbm_utils import LightGBMOptimizer, LightGBMSmallOptimizer
from xgboost_utils import XGBoostOptimizer, XGBoostSmallOptimizer
from catboost_utils import CatboostOptimizer, CatboostSmallOptimizer
from utils import FeatureGenerator, remove_seen, remove_useless_features, get_useless_columns, break_ties_with_filler
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

    def __init__(self, urms, ratings, validations, reference_baseline, n_folds=5, random_trials_perc=0.25):
        super(LightGBMTopkOptimizer, self).__init__(urms, ratings, validations,
                                                    n_folds=n_folds, random_trials_perc=random_trials_perc)
        self.reference_baseline = reference_baseline

    def evaluate(self, predictions_matrix, val, cutoff=10):
        return evaluate(fill_solution(predictions_matrix, self.reference_baseline, only_missing=True), val, cutoff=cutoff)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Final ensemble hyperparameter optimization')
    parser.add_argument('--ntrials', '-t', metavar='TRIALS', type=int, nargs='?', default=200,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--nfolds', '-cv', metavar='FOLDS', type=int, nargs='?', default=5,
                        help='Number of CV folds for hyperparameter optimization')
    parser.add_argument('--include-subs', '-i', nargs='?', dest="include_subs", default=False, const=True,
                        help='Whether to include good submissions in the ensemble')
    parser.add_argument('--force-hpo', '-f', nargs='?', dest="force_hpo", default=False, const=True,
                        help='Whether to run a new hyperparameter optimization discarding previous ones')
    parser.add_argument('--no-sub', '-ns', nargs='?', dest="no_sub", default=False, const=True,
                        help='Whether to avoid using this result for making a submission')
    parser.add_argument('--gbdt', '-g', metavar='GBDT', type=str, nargs='?', default='lightgbm', dest="gbdt",
                        choices=['lightgbm', 'xgboost', 'catboost'], help='GBDT model to use')
    parser.add_argument('--small', '-s', nargs='?', dest="small", default=False, const=True,
                        help='Whether to train on two different splittings, but on smaller train sets')

    args = parser.parse_args()
    n_trials = args.ntrials
    n_folds = args.nfolds
    force_hpo = args.force_hpo
    load_dataframe = True
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

        df_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + "final-ensemble-dataframe"
        basic_dfs = []

        if load_dataframe:
            for filename in sorted(glob.glob(df_filename + "*.parquet.gzip")):
                basic_dfs.append(pd.read_parquet(filename))

        featgen = FeatureGenerator(exam_folder)

        urms = featgen.get_urms()
        validations = featgen.get_validations()
        user_mappers = featgen.get_user_mappers()
        item_mappers = featgen.get_item_mappers()

        if len(basic_dfs) <= 0:

            all_predictions = []

            for folder in EXPERIMENTAL_CONFIG['datasets']:
                if exam_folder in folder:
                    print("Loading", folder)
                    all_predictions.append(
                        featgen.load_algorithms_predictions(folder, only_best_baselines=False, normalize=False))
                    all_predictions.append(featgen.load_folder_features(folder, include_fold_features=False))
                    for normalize in [True, False]:
                        featgen.load_ratings_ensemble_feature(folder, normalize=normalize)
                        featgen.load_lgbm_ensemble_feature(folder, normalize=normalize, break_ties=True)
                        featgen.load_lgbm_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                           algo_suffix="-small")
                        featgen.load_lgbm_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                           algo_suffix="-small-nonorm", featname_suffix="-nonorm")
                        featgen.load_lgbm_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                           algo_suffix="-small-both", featname_suffix="-both")
                        featgen.load_catboost_ensemble_feature(folder, normalize=normalize, break_ties=True)
                        featgen.load_xgb_ensemble_feature(folder, normalize=normalize, break_ties=True)
                        featgen.load_xgb_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                          algo_suffix="-small")
                        featgen.load_xgb_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                          algo_suffix="-small-nonorm", featname_suffix="-nonorm")
                        featgen.load_xgb_ensemble_feature(folder, normalize=normalize, break_ties=True,
                                                          algo_suffix="-small-both", featname_suffix="-both")

            basic_dfs = featgen.get_final_df()

            for predictions in all_predictions:
                for j in range(len(predictions)):
                    basic_dfs[j] = basic_dfs[j].merge(predictions[j], on=["user", "item"], how="left", sort=True)

            useless_cols = get_useless_columns(basic_dfs[-2])
            print("Useless columns: {}/{}".format(len(useless_cols), basic_dfs[-2].shape[1]))
            print(useless_cols)
            for i in range(len(basic_dfs)):
                remove_useless_features(basic_dfs[i], columns_to_remove=useless_cols, inplace=True)

            user_factors, item_factors = featgen.load_user_factors(exam_folder, num_factors=32, epochs=30, reg=5e-5,
                                                                   normalize=True, use_val_factors_for_test=False)
            for j in range(len(user_factors)):
                basic_dfs[j] = basic_dfs[j].merge(item_factors[j], on=["item"], how="left", sort=False)
                basic_dfs[j] = basic_dfs[j].merge(user_factors[j], on=["user"], how="left", sort=True)

            if args.include_subs:
                for sub in range(18, 25):
                    submission_name = "submission-{}".format(sub)
                    filename = "submission" + os.sep + "{}.zip".format(submission_name)
                    v, t = load_submission(filename, exam_folder, exam_user_mapper, exam_item_mapper)
                    basic_dfs[-2] = basic_dfs[-2].merge(pd.DataFrame({'user': v.row, 'item': v.col, submission_name: v.data}),
                                                        on=["user", "item"], how="left", sort=True)
                    basic_dfs[-1] = basic_dfs[-1].merge(pd.DataFrame({'user': t.row, 'item': t.col, submission_name: t.data}),
                                                        on=["user", "item"], how="left", sort=True)

            for j in range(len(basic_dfs)):
                basic_dfs[j].to_parquet(df_filename + "-{}.parquet.gzip".format(j), compression="gzip")

        print("Final dataframe shape:", basic_dfs[-2].shape)
        break_ties_folder = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        ratings_folder = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        fillers = None
        #[
        #    read_ratings(break_ties_folder + "valid_scores_ratings.tsv.gz".format(exam_folder), exam_user_mapper, exam_item_mapper),
        #    read_ratings(break_ties_folder + "test_scores_ratings.tsv.gz".format(exam_folder), exam_user_mapper, exam_item_mapper)
        #]
        optimizer = optimizer_class(urms, basic_dfs, validations, fillers=fillers, n_folds=n_folds, random_trials_perc=0.3)
        optimizer.optimize_all(exam_folder, force=force_hpo, n_trials=n_trials, folder=None,
                               study_name_suffix="_endtoend_" + args.gbdt)
        er, er_test, result = optimizer.train_cv_best_params(urms[-2], basic_dfs[-2], validations[-1], filler=None, test_df=basic_dfs[-1])
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

        filler = read_ratings(break_ties_folder + "valid_scores_ratings.tsv.gz".format(exam_folder), exam_user_mapper, exam_item_mapper)
        er = break_ties_with_filler(er, row_minmax_scaling(filler), use_filler_ratings=True, penalization=1e-6)

        filler = read_ratings(break_ties_folder + "test_scores_ratings.tsv.gz".format(exam_folder), exam_user_mapper, exam_item_mapper)
        er_test = break_ties_with_filler(er_test, row_minmax_scaling(filler), use_filler_ratings=True, penalization=1e-6)

        output_scores(ratings_folder + "valid_scores_{}.tsv".format(args.gbdt), er, user_mappers[-2], item_mappers[-2], compress=False)
        output_scores(ratings_folder + "test_scores_{}.tsv".format(args.gbdt), er_test, user_mappers[-1], item_mappers[-1], compress=False)

        if not args.no_sub:
            output_scores(ratings_folder + "valid_scores.tsv", er, user_mappers[-2], item_mappers[-2], compress=False)
            output_scores(ratings_folder + "test_scores.tsv", er_test, user_mappers[-1], item_mappers[-1], compress=False)

        scores, n_evals = evaluate(
            read_ratings(ratings_folder + "valid_scores_{}.tsv".format(args.gbdt), exam_user_mapper, exam_item_mapper),
            validations[-1]
        )
        print("FINAL ENSEMBLE {}: {:.8f}".format(exam_folder, np.sum(scores) / n_evals))

        del featgen
        del basic_dfs
        del validations
        del user_mappers
        del item_mappers
        del urms
        del optimizer

    if not args.no_sub:
        make_submission()

