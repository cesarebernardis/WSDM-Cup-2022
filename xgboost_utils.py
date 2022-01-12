import os
import optuna
import xgboost as xgb

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



class XGBoostOptimizer(Optimizer):

    NAME = "xgb"

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None, filler=None):

        # filler not supported

        model = xgb.XGBRanker(**_params)
        _ratings = _ratings.sort_values(by=['user'])

        validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
        validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1})
        validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True).fillna(0)

        _r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
        test_r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)

        splits = self.get_splits(_ratings, validation_df, random_state=103)
        for train_index, test_index in splits:

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
                eval_group=[test_groups], eval_metric="ndcg@10",
                callbacks=[xgb.callback.EarlyStopping(10)]
            )
            predictions = model.predict(train_df.iloc[test_index])
            predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                               _ratings.item.values[test_index])),
                                    shape=_validation.shape)

            _r += remove_seen(predictions_matrix, _urm)

            if test_df is not None:
                predictions = model.predict(test_df.drop(['user', 'item'], axis=1))
                predictions_matrix = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                                    shape=_validation.shape)
                predictions_matrix = remove_seen(predictions_matrix, _urm + _validation)
                test_r += predictions_matrix / len(splits)

        user_ndcg, n_evals = evaluate(_r, _validation, cutoff=10)
        _r = row_minmax_scaling(_r).tocoo()
        part_score = np.sum(user_ndcg) / n_evals

        if len(test_r.data) > 0:
            return _r, row_minmax_scaling(test_r).tocoo(), part_score

        return _r, part_score


    def get_params_from_trial(self, trial):
        return {
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.5, log=True),
            "objective": trial.suggest_categorical("objective", ["rank:ndcg", "rank:map"]),
            "gamma": trial.suggest_float("gamma", 1e-9, 1e-2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10., log=True),
            "subsample": trial.suggest_float("subsample", 1e-3, 1.0, log=True),
            "random_state": trial.suggest_categorical("random_state", [1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1., log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 10., log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 10., log=True),
            "base_score": trial.suggest_categorical("base_score", [0.]),
        }


class XGBoostSmallOptimizer(XGBoostOptimizer):

    NAME = "xgb-small"

    def get_splits(self, ratings, validation, random_state=1000):
        splits = []
        for i in range(2):
            splitter = RandomizedGroupKFold(self.n_folds, random_state=random_state + i)
            splits += [(tr_i, te_i) for tr_i, te_i in splitter.split(ratings, validation, ratings.user.values)]
        return splits

