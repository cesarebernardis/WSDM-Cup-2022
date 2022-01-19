import os
import optuna
import lightgbm as lgbm

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import RandomizedGroupKFold, Optimizer
from utils import remove_seen, row_minmax_scaling, evaluate


class LightGBMOptimizer(Optimizer):

    NAME = "lgbm"
    split_seed = 687

    def evaluate(self, predictions_matrix, val, cutoff=10):
        return evaluate(predictions_matrix, val, cutoff=cutoff)

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None, filler=None):

        if "n_estimators" not in _params.keys():
            _params["n_estimators"] = 1000
        _params["metric"] = None
        #_params["n_jobs"] = 4
        model = lgbm.LGBMRanker(**_params)
        _ratings = _ratings.sort_values(by=['user'])

        validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
        validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1 if filler is None else 10})
        validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True)

        train_validation_df = validation_df.copy()
        validation_df.fillna(0, inplace=True)

        if filler is None:
            train_validation_df.fillna(0, inplace=True)
        else:
            filler = row_minmax_scaling(filler).tocoo()
            validation_filler = pd.DataFrame({'user': filler.row, 'item': filler.col, 'filler': (filler.data * 10).astype(np.int32)})
            train_validation_df = train_validation_df.merge(validation_filler, on=["user", "item"], how="left", sort=True)
            train_validation_df.filler.fillna(0., inplace=True)
            train_validation_df.relevance.fillna(train_validation_df.filler, inplace=True)
            train_validation_df.drop(['filler'], axis=1, inplace=True)

        _r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
        test_r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)

        splits = self.get_splits(_ratings, validation_df, random_state=13)
        for train_index, test_index in splits:

            # Ensure to keep order
            train_index = np.sort(train_index)
            test_index = np.sort(test_index)

            users, counts = np.unique(_ratings.user.values[train_index], return_counts=True)
            train_groups = counts[np.argsort(users)]

            users, counts = np.unique(_ratings.user.values[test_index], return_counts=True)
            test_groups = counts[np.argsort(users)]

            train_df = _ratings.drop(['user', 'item'], axis=1)
            train_pred_df = train_validation_df.sort_values(by=['user']).drop(['user', 'item'], axis=1)
            val_pred_df = validation_df.sort_values(by=['user']).drop(['user', 'item'], axis=1)
            model.fit(
                train_df.iloc[train_index], train_pred_df.iloc[train_index], group=train_groups,
                eval_set=[(train_df.iloc[test_index], val_pred_df.iloc[test_index])],
                eval_group=[test_groups], eval_at=(1, 10), eval_metric="ndcg@10",
                callbacks=[lgbm.early_stopping(10, verbose=False), lgbm.log_evaluation(0)]
            )
            predictions = model.predict(train_df.iloc[test_index], raw_score=True) + 1e-10
            predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                               _ratings.item.values[test_index])),
                                    shape=_validation.shape)
            _r += remove_seen(predictions_matrix, _urm) / int(len(splits) / self.n_folds)

            if test_df is not None:
                predictions = model.predict(test_df.drop(['user', 'item'], axis=1), raw_score=True) + 1e-10
                predictions_matrix = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                                    shape=_validation.shape)
                predictions_matrix = remove_seen(predictions_matrix, _urm + _validation)
                test_r += predictions_matrix / len(splits)

        user_ndcg, n_evals = self.evaluate(_r, _validation, cutoff=10)
        part_score = np.sum(user_ndcg) / n_evals
        _r = _r.tocoo()

        if test_df is not None:
            return _r, test_r.tocoo(), part_score

        return _r, part_score


    def get_params_from_trial(self, trial):

        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])

        # https://github.com/Microsoft/LightGBM/issues/695#issuecomment-315591634
        params = {
            "boosting_type": boosting_type,
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.4, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 47),
            "num_leaves": trial.suggest_int("num_leaves", 7, 2047),
            "subsample_for_bin": trial.suggest_int("subsample_for_bin", 1000, 1000000),
            "objective": trial.suggest_categorical("objective", ["lambdarank", "rank_xendcg"]),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1e-2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 8, 100),
            "random_state": trial.suggest_categorical("random_state", [1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-2, 1., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 0.01, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 0.01, log=True),
        }

        if boosting_type != "goss":
            params["subsample_freq"] = trial.suggest_int("subsample_freq", 0, 5)
            if params["subsample_freq"] > 0:
                params["subsample"] = trial.suggest_float("subsample", 1e-2, 1., log=True)
            else:
                params["subsample"] = 1.

        if boosting_type == "dart":
            params["n_estimators"] = trial.suggest_int("n_estimators", 400, 1200, log=True)
            params["drop_rate"] = trial.suggest_float("drop_rate", 1e-2, 0.5, log=True)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.2, 0.8)
            params["max_drop"] = trial.suggest_int("max_drop", 5, 100, log=True)
        elif boosting_type == "goss":
            params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.3)
            params["other_rate"] = trial.suggest_float("other_rate", 0.02, 0.2)

        return params



class LightGBMSmallOptimizer(LightGBMOptimizer):

    NAME = "lgbm-small"

    def get_splits(self, ratings, validation, random_state=1000):
        splits = []
        for i in range(2):
            splitter = RandomizedGroupKFold(self.n_folds, random_state=random_state + i)
            splits += [(tr_i, te_i) for tr_i, te_i in splitter.split(ratings, validation, ratings.user.values)]
        return splits

