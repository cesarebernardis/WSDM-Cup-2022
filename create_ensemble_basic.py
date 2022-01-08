import os

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, make_submission, first_level_ensemble



if __name__ == "__main__":

    output_sub = True
    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])
    cold_user_threshold = EXPERIMENTAL_CONFIG['cold_user_threshold']
    quite_cold_user_threshold = EXPERIMENTAL_CONFIG['quite_cold_user_threshold']
    quite_warm_user_threshold = EXPERIMENTAL_CONFIG['quite_warm_user_threshold']

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, _, _ = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
        exam_folder_profile_lengths = np.ediff1d(exam_train.get_URM().indptr)
        itempop = np.ediff1d(exam_train.get_URM().tocsc().indptr)

        user_masks = {
            "cold": exam_folder_profile_lengths < cold_user_threshold,
            "quite-cold": np.logical_and(exam_folder_profile_lengths >= cold_user_threshold,
                                         exam_folder_profile_lengths < quite_cold_user_threshold),
            "quite-warm": np.logical_and(exam_folder_profile_lengths >= quite_cold_user_threshold,
                                         exam_folder_profile_lengths < quite_warm_user_threshold),
            "warm": exam_folder_profile_lengths >= quite_warm_user_threshold,
        }

        with open(EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + "best-last-level-ensemble-weights.pkl", "rb") as file:
            ensemble_weights = pkl.load(file)

        urm_exam_valid_total = None
        urm_exam_test_total = None

        folder_keys = set(k for k in ensemble_weights['cold'].keys() if "sign" not in k and "exp" not in k)

        for folder in folder_keys:

            urm_valid_total, urm_test_total = first_level_ensemble(
                folder, exam_folder, exam_valid, exam_folder_profile_lengths, user_masks)

            folder_train, _, _, _ = create_dataset_from_folder(folder)
            user_mapper, item_mapper = folder_train.get_URM_mapper()

            inv_user_mapper = {v: k for k, v in exam_user_mapper.items()}
            um = - np.ones(len(exam_folder_profile_lengths), dtype=np.int32)
            for k in inv_user_mapper.keys():
                um[k] = user_mapper[inv_user_mapper[k]]
            profile_lengths_diff = np.ediff1d(folder_train.get_URM().indptr)[um[um >= 0]] - exam_folder_profile_lengths

            algo_weights = np.zeros(urm_valid_total.shape[0], dtype=np.float32)
            for usertype in ["cold", "quite-cold", "quite-warm", "warm"]:
                mask =  user_masks[usertype]
                algo_weights[mask] = ensemble_weights[usertype][folder] * ensemble_weights[usertype][folder + "-sign"]
                #exponent = ensemble_weights[usertype][folder + "-pldiff-exp"] * \
                #           ensemble_weights[usertype][folder + "-pldiff-exp-sign"]
                #algo_weights[mask] = np.multiply(algo_weights[mask], np.power(profile_lengths_diff[mask] + 1., exponent))

            weights = sps.diags(algo_weights, dtype=np.float32)
            urm_valid_total = weights.dot(row_minmax_scaling(urm_valid_total))

            user_ndcg, n_evals = evaluate(urm_valid_total, exam_valid.get_URM(), cutoff=10)
            avg_ndcg = np.sum(user_ndcg) / n_evals
            outstr = "\t".join(map(str, ["Validation", exam_folder, folder, avg_ndcg]))
            print(outstr)

            if urm_exam_valid_total is None:
                urm_exam_valid_total = urm_valid_total
            else:
                urm_exam_valid_total += urm_valid_total

            urm_test_total = weights.dot(row_minmax_scaling(urm_test_total))

            if urm_exam_test_total is None:
                urm_exam_test_total = urm_test_total
            else:
                urm_exam_test_total += urm_test_total

        urm_valid_total.eliminate_zeros()
        urm_test_total.eliminate_zeros()

        user_ndcg, n_evals = evaluate(urm_exam_valid_total, exam_valid.get_URM(), cutoff=10)
        avg_ndcg = np.sum(user_ndcg) / n_evals
        outstr = "\t".join(map(str, ["Validation", exam_folder, "Ensemble", avg_ndcg]))
        print(outstr)

        basepath = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        output_scores(basepath + "valid_scores_ratings.tsv.gz", urm_exam_valid_total, exam_user_mapper, exam_item_mapper, compress=False)
        output_scores(basepath + "test_scores_ratings.tsv.gz", urm_exam_test_total, exam_user_mapper, exam_item_mapper, compress=False)

        if output_sub:
            output_scores(basepath + "valid_scores.tsv", urm_exam_valid_total, exam_user_mapper, exam_item_mapper, compress=False)
            output_scores(basepath + "test_scores.tsv", urm_exam_test_total, exam_user_mapper, exam_item_mapper, compress=False)

        print("----------------------------------")

    if output_sub:
        make_submission()

