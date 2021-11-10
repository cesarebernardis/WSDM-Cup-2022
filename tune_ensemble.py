import os
import optuna

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble


EXPERIMENTAL_CONFIG['n_folds'] = 0

if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])
    n_trials = 500
    val_weight = 2.
    cold_user_threshold = EXPERIMENTAL_CONFIG['cold_user_threshold']
    quite_cold_user_threshold = EXPERIMENTAL_CONFIG['quite_cold_user_threshold']
    quite_warm_user_threshold = EXPERIMENTAL_CONFIG['quite_warm_user_threshold']

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()

        validations = []
        exam_profile_lengths = []
        user_mappers = []
        item_mappers = []
        user_masks = dict((k, []) for k in ['cold', 'quite-cold', 'quite-warm', 'warm'])
        for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
            validation_path = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + exam_folder + "-" + str(fold) + os.sep
            with open(validation_path + "URM_all_train_mapper", "rb") as file:
                um, im = pkl.load(file)
                user_mappers.append(um)
                item_mappers.append(im)
            urm_train = sps.load_npz(validation_path + "URM_all_train.npz")
            exam_profile_lengths.append(np.ediff1d(urm_train.indptr))
            user_masks['cold'].append(exam_profile_lengths[-1] < cold_user_threshold)
            user_masks['quite-cold'].append(np.logical_and(exam_profile_lengths[-1] >= cold_user_threshold,
                                                           exam_profile_lengths[-1] < quite_cold_user_threshold))
            user_masks['quite-warm'].append(np.logical_and(exam_profile_lengths[-1] >= quite_cold_user_threshold,
                                                           exam_profile_lengths[-1] < quite_warm_user_threshold))
            user_masks['warm'].append(exam_profile_lengths[-1] >= quite_warm_user_threshold)
            validations.append(sps.load_npz(validation_path + "URM_all_test.npz"))

        user_mappers.append(exam_user_mapper)
        item_mappers.append(exam_item_mapper)
        exam_profile_lengths.append(np.ediff1d(exam_train.get_URM().indptr))
        user_masks['cold'].append(exam_profile_lengths[-1] < cold_user_threshold)
        user_masks['quite-cold'].append(np.logical_and(exam_profile_lengths[-1] >= cold_user_threshold,
                                                       exam_profile_lengths[-1] < quite_cold_user_threshold))
        user_masks['quite-warm'].append(np.logical_and(exam_profile_lengths[-1] >= quite_cold_user_threshold,
                                                       exam_profile_lengths[-1] < quite_warm_user_threshold))
        user_masks['warm'].append(exam_profile_lengths[-1] >= quite_warm_user_threshold)
        validations.append(exam_valid.get_URM())

        def _evaluate(_validation, _ratings, _weights):
            _r = None
            for name, weights in _weights.items():
                if _r is None:
                    _r = weights.dot(_ratings[name])
                else:
                    _r += weights.dot(_ratings[name])
            user_ndcg, n_evals = evaluate(_r, _validation, cutoff=10)
            return np.sum(user_ndcg) / n_evals

        def objective(trial, _ratings, _user_masks, _profile_lengths_diff=None):

            _pl_weights = {}
            _exponents = {}

            for recname in _ratings[-1].keys():
                _pl_weights[recname] = trial.suggest_float(recname, 1e-5, 1., log=True) * \
                                       trial.suggest_categorical(recname + "-sign", [-1., 1.])
                if _profile_lengths_diff is not None:
                    _exponents[recname] = trial.suggest_float(recname + "-pldiff-exp", 1e-4, 1., log=True) * \
                                          trial.suggest_categorical(recname + "-pldiff-exp-sign", [-1., 1.])

            _val_weight = val_weight
            final_score = 0.
            denominator = 0.

            for fold in range(-1, EXPERIMENTAL_CONFIG['n_folds']):

                n_users = len(_user_masks[fold])
                weights = np.zeros(n_users, dtype=np.float32)
                weights[_user_masks[fold]] = 1.
                val = sps.diags(weights, dtype=np.float32).dot(validations[fold])
                val.eliminate_zeros()

                _weights = {}
                for recname in _ratings[fold].keys():
                    weights = np.zeros(n_users, dtype=np.float32)
                    weights[_user_masks[fold]] = _pl_weights[recname]
                    if _profile_lengths_diff is not None:
                        weights = np.multiply(weights, np.power(_profile_lengths_diff[fold][recname] + 1.,
                                                                _exponents[recname]))
                    _weights[recname] = sps.diags(weights, dtype=np.float32)

                final_score += _evaluate(val, _ratings[fold], _weights) * _val_weight
                denominator += _val_weight
                _val_weight = 1.

            return final_score / denominator

        ensemble_ratings = {}
        profile_lengths_diff = [{} for _ in range(-1, EXPERIMENTAL_CONFIG['n_folds'])]
        run_last_level_opt = False

        for folder in EXPERIMENTAL_CONFIG['datasets']:

            if exam_folder in folder:

                folder_train, _, _, _ = create_dataset_from_folder(folder)
                user_mapper, item_mapper = folder_train.get_URM_mapper()
                fpl = np.ediff1d(folder_train.get_URM().indptr)

                for i, epl in enumerate(exam_profile_lengths):
                    inv_user_mapper = {v: k for k, v in user_mappers[i].items()}
                    um = - np.ones(len(epl), dtype=np.int32)
                    for k in inv_user_mapper.keys():
                        um[k] = user_mapper[inv_user_mapper[k]]
                    profile_lengths_diff[i][folder] = fpl[um[um >= 0]] - epl

                ratings = [{} for _ in range(EXPERIMENTAL_CONFIG['n_folds'] + 1)]
                for algorithm in EXPERIMENTAL_CONFIG['baselines']:
                    is_complete = True
                    recname = algorithm.RECOMMENDER_NAME
                    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
                    print("Loading", folder, recname)
                    for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                        validation_folder = exam_folder + "-" + str(fold)
                        #with open("{}URM_all_train_mapper".format(EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + validation_folder + os.sep), "rb") as file:
                        #    user_mapper, item_mapper = pkl.load(file)
                        try:
                            ratings[fold][recname] = row_minmax_scaling(
                                read_ratings(output_folder_path + validation_folder + "_valid_scores.tsv",
                                            exam_user_mapper, exam_item_mapper))
                        except Exception as e:
                            is_complete = False
                            continue
                    try:
                        ratings[-1][recname] = row_minmax_scaling(
                            read_ratings(output_folder_path + exam_folder + "_valid_scores.tsv",
                                        exam_user_mapper, exam_item_mapper))
                    except Exception as e:
                        is_complete = False
                        continue
                    if not is_complete:
                        for fold in range(EXPERIMENTAL_CONFIG['n_folds'] + 1):
                            del ratings[fold][recname]

                run_opt = False
                weights_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + "best-ensemble-weights.pkl"
                if os.path.isfile(weights_filename):
                    with open(weights_filename, "rb") as file:
                        result_dict = pkl.load(file)
                    for k in result_dict['cold'].keys():
                        if k not in ratings[-1].keys():
                            run_opt = True
                else:
                    run_opt = True

                if run_opt:
                    run_last_level_opt = True
                    result_dict = {}
                    print("Starting cold optimization")
                    storage = "sqlite:///" + EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + "optuna.db"
                    study_name = "base-ensemble-cold-{}-{}".format(folder, exam_folder)
                    optuna.study.delete_study(study_name, storage)
                    _objective = lambda x: objective(x, ratings, user_masks['cold'])
                    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
                    study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
                    result_dict["cold"] = study.best_params

                    print("Starting quite-cold optimization")
                    study_name = "base-ensemble-quitecold-{}-{}".format(folder, exam_folder)
                    optuna.study.delete_study(study_name, storage)
                    _objective = lambda x: objective(x, ratings, user_masks['quite-cold'])
                    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
                    study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
                    result_dict["quite-cold"] = study.best_params

                    print("Starting quite-warm optimization")
                    study_name = "base-ensemble-quitewarm-{}-{}".format(folder, exam_folder)
                    optuna.study.delete_study(study_name, storage)
                    _objective = lambda x: objective(x, ratings, user_masks['quite-warm'])
                    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
                    study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
                    result_dict["quite-warm"] = study.best_params

                    print("Starting warm optimization")
                    study_name = "base-ensemble-warm-{}-{}".format(folder, exam_folder)
                    optuna.study.delete_study(study_name, storage)
                    _objective = lambda x: objective(x, ratings, user_masks['warm'])
                    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
                    study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
                    result_dict["warm"] = study.best_params

                    with open(weights_filename, "wb") as file:
                        pkl.dump(result_dict, file)

                ensemble_ratings[folder], _ = first_level_ensemble(folder, exam_folder, exam_valid,
                                exam_profile_lengths[-1], dict((k, user_masks[k][-1]) for k in user_masks.keys()))

        weights_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + "best-last-level-ensemble-weights.pkl"
        if os.path.isfile(weights_filename):
            with open(weights_filename, "rb") as file:
                result_dict = pkl.load(file)
            for k in result_dict['cold'].keys():
                if k not in ensemble_ratings.keys():
                    run_last_level_opt = True
        else:
            run_last_level_opt = True

        if run_last_level_opt:

            result_dict = {}
            print("Starting cold optimization")
            study_name = "last-level-ensemble-cold-{}".format(exam_folder)
            storage = "sqlite:///" + EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + "optuna.db"
            optuna.study.delete_study(study_name, storage)
            _objective = lambda x: objective(x, [ensemble_ratings], user_masks['cold'], profile_lengths_diff)
            study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
            result_dict["cold"] = study.best_params

            print("Starting quite-cold optimization")
            study_name = "last-level-ensemble-quitecold-{}".format(exam_folder)
            optuna.study.delete_study(study_name, storage)
            _objective = lambda x: objective(x, [ensemble_ratings], user_masks['quite-cold'], profile_lengths_diff)
            study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
            result_dict["quite-cold"] = study.best_params

            print("Starting quite-warm optimization")
            study_name = "last-level-ensemble-quitewarm-{}".format(exam_folder)
            optuna.study.delete_study(study_name, storage)
            _objective = lambda x: objective(x, [ensemble_ratings], user_masks['quite-warm'], profile_lengths_diff)
            study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
            result_dict["quite-warm"] = study.best_params

            print("Starting warm optimization")
            study_name = "last-level-ensemble-warm-{}".format(exam_folder)
            optuna.study.delete_study(study_name, storage)
            _objective = lambda x: objective(x, [ensemble_ratings], user_masks['warm'], profile_lengths_diff)
            study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(_objective, n_trials=max(0, n_trials - len(study.trials)), show_progress_bar=True)
            result_dict["warm"] = study.best_params

            with open(weights_filename, "wb") as file:
                pkl.dump(result_dict, file)
