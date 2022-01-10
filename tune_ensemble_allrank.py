import os
import optuna
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl
import tempfile

from sklearn.datasets import dump_svmlight_file

from urllib.parse import urlparse

import allrank.models.losses as losses
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank_utils import fit, predict
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim

from utils import RandomizedGroupKFold

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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class AllRankOptimizer(Optimizer):

    NAME = "allrank"

    def __init__(self, urms, ratings, validations, fillers=None, n_folds=5, random_trials_perc=0.25, with_transformer=True):
        super(AllRankOptimizer, self).__init__(urms, ratings, validations, fillers=fillers, n_folds=n_folds, random_trials_perc=random_trials_perc)
        self.with_transformer = with_transformer
        self.tempdir = tempfile.TemporaryDirectory()
        self.input_path = self.tempdir + os.sep

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None, filler=None):

        config = {}
        config["training"] = {
            "epochs": 200,
            "early_stopping_patience": 5,
            "gradient_clipping_norm": null
        }
        config["val_metric"] = "ndcg_10"
        config["metrics"] = ["ndcg_5", "ndcg_10", "ndcg_30"]
        config["detect_anomaly"] = False,
        config["expected_metrics"] = { "val": { "ndcg_10": 0.5 } }

        fc_model_sizes = []
        for k in sorted(_params.keys()):
            if "fc_model_sizes" in k:
                fc_model_sizes.append(_params[k])

        config["model"] = {
            "fc_model": {
                "sizes": fc_model_sizes,
                "input_norm": _params["input_norm"],
                "activation": _params["activation"],
                "dropout": _params["dropout"],
            },
            "post_model": {
                "output_activation": _params["post_activation"],
                "d_output": 100
            }
        }
        if self.with_transformer:
            config["model"]["transformer"] = {
                "N": _params["transformer_N"],
                "d_ff": _params["transformer_dff"],
                "h": _params["transformer_heads"],
                "positional_encoding": None,
                "dropout": _params["transformer_dropout"],
            }

        config["optimizer"] = {
            "name": "Adam",
            "args": {"lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)}
        }
        config["lr_scheduler"] = {"name": "StepLR", "args": {"step_size": 50, "gamma": 0.1}}
        config["loss"] = {
            "name": _params["loss"],
            "args": dict((k[10:], _params[k]) for k in _params.keys() if k.beginswith("loss_args"))
        }

        config = dotdict(config)

        splitter = RandomizedGroupKFold(self.n_folds, random_state=42)
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

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)

        if test_df is not None:
            if not os.path.isfile(self.input_path + str(i) + os.sep + "test.txt"):
                dump_svmlight_file(test_df.drop(['user', 'item'], axis=1), np.zeros(len(test_df), dtype=np.float32),
                                   self.input_path + str(i) + os.sep + "test.txt", query_id=test_df.user)

        for i, (train_index, test_index) in enumerate(splitter.split(_ratings, validation_df, _ratings.user.values)):

            if not (os.path.isfile(self.input_path + str(i) + os.sep + "valid.txt") and \
                    os.path.isfile(self.input_path + str(i) + os.sep + "train.txt")):

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

                dump_svmlight_file(train_df.iloc[train_index], train_pred_df.iloc[train_index],
                                   self.input_path + str(i) + os.sep + "train.txt",
                                   query_id=_ratings.user[train_index])

                dump_svmlight_file(train_df.iloc[test_index], val_pred_df.iloc[test_index],
                                   self.input_path + str(i) + os.sep + "valid.txt",
                                   query_id=_ratings.user[test_index])

            # train_ds, val_ds
            train_ds, val_ds = load_libsvm_dataset(
                input_path=self.input_path + str(i),
                slate_length=100,
                validation_ds_role="vali",
            )

            n_features = train_ds.shape[-1]
            assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

            # train_dl, val_dl
            train_dl, val_dl = create_data_loaders(
                train_ds, val_ds, num_workers=2, batch_size=_params["batch_size"])

            # gpu support
            dev = get_torch_device()
            logger.info("Model training will execute on {}".format(dev.type))

            # instantiate model
            model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
            if torch.cuda.device_count() > 1:
                model = CustomDataParallel(model)
                logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
            model.to(dev)

            # load optimizer, loss and LR scheduler
            optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
            loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
            if config.lr_scheduler.name:
                scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
            else:
                scheduler = None

            with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
                # run training
                result = fit(
                    model=model,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_dl=train_dl,
                    valid_dl=val_dl,
                    config=config,
                    device=dev,
                    **asdict(config.training)
                )

            predictions = predict(val_ds, model) + 1e-10
            predictions_matrix = sps.csr_matrix((predictions, (_ratings.user.values[test_index],
                                                               _ratings.item.values[test_index])),
                                    shape=_validation.shape)
            _r += remove_seen(predictions_matrix, _urm)

            if test_df is not None:
                _, test_ds = load_libsvm_dataset(
                    input_path=self.input_path + str(i),
                    slate_length=100,
                    validation_ds_role="test",
                )
                predictions = predict(test_ds, model) + 1e-10
                predictions_matrix = sps.csr_matrix((predictions, (test_df.user.values, test_df.item.values)),
                                                    shape=_validation.shape)
                predictions_matrix = remove_seen(predictions_matrix, _urm + _validation)
                test_r += predictions_matrix / splitter.get_n_splits()

            model.apply(weight_reset)

        user_ndcg, n_evals = evaluate(_r, _validation, cutoff=10)
        _r = row_minmax_scaling(_r).tocoo()
        part_score = np.sum(user_ndcg) / n_evals

        if test_df is not None:
            return _r, row_minmax_scaling(test_r).tocoo(), part_score

        return _r, part_score


    def get_params_from_trial(self, trial):

        params = {
            "input_norm": trial.suggest_categorical("input_norm", [True, False]),
            "activation": trial.suggest_categorical("activation", ["ReLU", "ELU", "SELU"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "batch_size": trial.suggest_int("batch_size", 16, 256, log=True),
            "post_activation": trial.suggest_categorical("post_activation", ["SELU", "ReLU", "Sigmoid", "TanH"]),
        }

        n_layers = trial.suggest_int("n_layers", 1, 4)
        for i in range(n_layers):
            params["fc_model_sizes_{}".format(i)] = trial.suggest_int("batch_size", 16, 256, log=True),

        if self.with_transformer:
            params["transformer_N"] = trial.suggest_int("transformer_N", 1, 4),
            params["transformer_dff"] = trial.suggest_int("transformer_dff", 32, 256, log=True),
            params["transformer_heads"] = trial.suggest_int("transformer_heads", 2, 8),
            params["transformer_dropout"] = trial.suggest_float("transformer_dropout", 0.0, 0.6, log=False),

        return params




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LightGBM ensemble hyperparameter optimization')
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

                useless_cols = get_useless_columns(ratings[-2])
                for i in range(len(ratings)):
                    remove_useless_features(ratings[i], columns_to_remove=useless_cols, inplace=True)

                user_factors, item_factors = featgen.load_user_factors(folder, num_factors=12, normalize=True)
                for j in range(len(user_factors)):
                    ratings[j] = ratings[j].merge(item_factors[j], on=["item"], how="left", sort=False)
                    ratings[j] = ratings[j].merge(user_factors[j], on=["user"], how="left", sort=True)

                optimizer = AllRankOptimizer(urms, ratings, validations, n_folds=n_folds)
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
