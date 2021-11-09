import os

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, make_submission



if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])
    ndcg_power = 2.
    cold_user_threshold = 4

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        results_filename = exam_train.get_complete_folder() + os.sep + "results.tsv"
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
        exam_folder_profile_lengths = np.ediff1d(exam_train.get_URM().indptr)
        urm_exam_valid_total = None
        urm_exam_test_total = None
        exam_valid_users_to_evaluate = np.arange(urm_exam_valid_neg.shape[0])[np.ediff1d(urm_exam_valid_neg.indptr) > 0]
        exam_test_users_to_evaluate = np.arange(urm_exam_test_neg.shape[0])[np.ediff1d(urm_exam_test_neg.indptr) > 0]

        for folder in EXPERIMENTAL_CONFIG['datasets']:

            urm_valid_total = None
            urm_test_total = None

            if exam_folder in folder:

                train, _, _, _ = create_dataset_from_folder(folder)
                user_mapper, item_mapper = train.get_URM_mapper()
                folder_urm = compress_urm(train.get_URM(), user_mapper, item_mapper, exam_user_mapper, exam_item_mapper)
                folder_profile_lengths = np.ediff1d(folder_urm.indptr)

                for algorithm in EXPERIMENTAL_CONFIG['baselines']:

                    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + algorithm.RECOMMENDER_NAME + os.sep
                    try:
                        ratings = read_ratings(output_folder_path + exam_folder + "_valid_scores.tsv", exam_user_mapper, exam_item_mapper)
                    except Exception as e:
                        continue

                    ratings = row_minmax_scaling(ratings)
                    user_ndcg, n_evals = evaluate(ratings, exam_valid.get_URM(), cutoff=10)
                    avg_ndcg = np.sum(user_ndcg) / n_evals
                    outstr = "\t".join(map(str, ["Validation", exam_folder, folder, algorithm.RECOMMENDER_NAME, avg_ndcg]))
                    with open(results_filename, "a") as file:
                        print(outstr, file=file)
                        print(outstr)

                    # Might be overfitting
                    weights = np.power(user_ndcg, ndcg_power) #+ avg_ndcg ** ndcg_power
                    ratings = sps.diags(weights, dtype=np.float32).dot(ratings)

                    if urm_valid_total is None:
                        urm_valid_total = ratings
                    else:
                        urm_valid_total += ratings

                    ratings = read_ratings(output_folder_path + exam_folder + "_test_scores.tsv", exam_user_mapper, exam_item_mapper)
                    ratings = row_minmax_scaling(ratings)
                    ratings = sps.diags(weights, dtype=np.float32).dot(ratings)

                    if urm_test_total is None:
                        urm_test_total = ratings
                    else:
                        urm_test_total += ratings

                if urm_valid_total is None:
                    continue

                urm_valid_total = row_minmax_scaling(urm_valid_total)
                user_ndcg, n_evals = evaluate(urm_valid_total, exam_valid.get_URM(), cutoff=100)
                avg_ndcg = np.sum(user_ndcg) / n_evals
                outstr = "\t".join(map(str, ["Validation", exam_folder, folder, "First level Ensemble", avg_ndcg]))
                with open(results_filename, "a") as file:
                    print(outstr, file=file)
                    print(outstr)

                # Might be overfitting
                weights = np.power(user_ndcg, ndcg_power) #+ avg_ndcg ** ndcg_power
                if folder != exam_folder:
                    mask = np.logical_and(exam_folder_profile_lengths < cold_user_threshold,
                                          folder_profile_lengths >= cold_user_threshold)
                    weights = np.multiply(weights, mask.astype(np.float32))
                urm_valid_total = sps.diags(weights, dtype=np.float32).dot(urm_valid_total)

                if urm_exam_valid_total is None:
                    urm_exam_valid_total = urm_valid_total
                else:
                    urm_exam_valid_total += urm_valid_total

                urm_test_total = row_minmax_scaling(urm_test_total)
                urm_test_total = sps.diags(weights, dtype=np.float32).dot(urm_test_total)

                if urm_exam_test_total is None:
                    urm_exam_test_total = urm_test_total
                else:
                    urm_exam_test_total += urm_test_total

        user_ndcg, n_evals = evaluate(urm_exam_valid_total, exam_valid.get_URM(), cutoff=10)
        avg_ndcg = np.sum(user_ndcg) / n_evals
        outstr = "\t".join(map(str, ["Validation", exam_folder, "None", "Last level Ensemble", avg_ndcg]))
        with open(results_filename, "a") as file:
            print(outstr, file=file)
            print(outstr)

        basepath = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        output_scores(basepath + "valid_scores.tsv", urm_exam_valid_total, exam_user_mapper, exam_item_mapper, user_prefix=exam_folder)
        output_scores(basepath + "test_scores.tsv", urm_exam_test_total, exam_user_mapper, exam_item_mapper, user_prefix=exam_folder)

        print("----------------------------------")

    make_submission()

