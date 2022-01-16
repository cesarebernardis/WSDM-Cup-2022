import os
import optuna
import catboost

import numpy as np
import pandas as pd
import scipy.sparse as sps

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import RandomizedGroupKFold, Optimizer
from utils import remove_seen, row_minmax_scaling, evaluate


class CatboostOptimizer(Optimizer):

    NAME = "catboost"

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None, filler=None):

        # filler not supported
        #_params["thread_count"] = 14

        model = catboost.CatBoostRanker(**_params)
        _ratings = _ratings.sort_values(by=['user'])

        validation_df = pd.DataFrame({'user': _ratings.user.copy(), 'item': _ratings.item.copy()})
        validation_true = pd.DataFrame({'user': _validation.row, 'item': _validation.col, 'relevance': 1})
        validation_df = validation_df.merge(validation_true, on=["user", "item"], how="left", sort=True).fillna(0)

        _r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)
        test_r = sps.csr_matrix(([], ([], [])), shape=_validation.shape)

        splits = self.get_splits(_ratings, validation_df, random_state=1003)
        for train_index, test_index in splits:

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
            _r += remove_seen(predictions_matrix, _urm) / int(len(splits) / self.n_folds)

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
                test_r += predictions_matrix / len(splits)

        user_ndcg, n_evals = evaluate(_r, _validation, cutoff=10)
        part_score = np.sum(user_ndcg) / n_evals
        _r = row_minmax_scaling(_r).tocoo()

        if len(test_r.data) > 0:
            return _r, row_minmax_scaling(test_r).tocoo(), part_score

        return _r, part_score


    def get_params_from_trial(self, trial):

        bagging_temperature = None
        subsample = None
        grow_policy = None  # trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])
        objective = trial.suggest_categorical("objective", ["YetiRank", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE"])

        sampling_unit = None
        if objective == "YetiRankPairwise":
            sampling_unit = trial.suggest_categorical("sampling_unit", ["Object", "Group"])

        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"])

        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 1e-2, 100., log=True)
        else:
            subsample = trial.suggest_float("subsample", 1e-2, 0.7, log=True)

        max_max_depth = 12
        if objective in ["YetiRank", "YetiRankPairwise"]:
            max_max_depth = 8

        params = {
            "objective": objective,
            "custom_metric": trial.suggest_categorical("custom_metric", ["NDCG:top=10"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["NDCG:top=10"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.4, log=True),
            "grow_policy": grow_policy,
            "iterations": trial.suggest_categorical("iterations", [2000]),
            "task_type": trial.suggest_categorical("task_type", ["GPU"]),
            "max_depth": trial.suggest_int("max_depth", 2, max_max_depth),
            "random_state": trial.suggest_categorical("random_state", [1]),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10., log=True),
            "bootstrap_type": bootstrap_type,
            "subsample": subsample,
            "bagging_temperature": bagging_temperature,
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1., log=True),
            "score_function": trial.suggest_categorical("score_function", ["Cosine", "L2"]),
            "sampling_unit": sampling_unit,
        }

        return params


class CatboostSmallOptimizer(CatboostOptimizer):

    NAME = "catboost-small"

    def get_splits(self, ratings, validation, random_state=1000):
        splits = []
        for i in range(2):
            splitter = RandomizedGroupKFold(self.n_folds, random_state=random_state + i)
            splits += [(tr_i, te_i) for tr_i, te_i in splitter.split(ratings, validation, ratings.user.values)]
        return splits


