import os

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import create_dataset_from_folder, compress_urm


if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])

    for folder in EXPERIMENTAL_CONFIG['datasets']:

        train, valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(folder)
        user_mapper, item_mapper = train.get_URM_mapper()
        profile_lengths = np.ediff1d(train.get_URM().indptr)
        print(folder, len(user_mapper), len(item_mapper))

        for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

            if exam_folder == folder:
                continue

            exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
            exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
            exam_profile_lengths = np.ediff1d(exam_train.get_URM().indptr)

            inv_exam_user_mapper = {v: k for k, v in exam_user_mapper.items()}
            inv_exam_item_mapper = {v: k for k, v in exam_item_mapper.items()}

            print("Validation", folder, exam_folder)

            counter = 0
            counter_better = 0
            counter_best = 0
            users_to_recommend = np.arange(exam_train.n_users)[np.ediff1d(urm_exam_valid_neg.indptr) > 0]
            for u in users_to_recommend:
                if inv_exam_user_mapper[u] in user_mapper.keys():
                    exam_user_in_dataset = user_mapper[inv_exam_user_mapper[u]]
                    counter += 1
                    if exam_profile_lengths[u] < profile_lengths[exam_user_in_dataset]:
                        counter_better += 1
                        if exam_profile_lengths[u] < 4:
                            counter_best += 1
                    #print(inv_exam_user_mapper[u], exam_profile_lengths[u], profile_lengths[exam_user_in_dataset])
            print("{}/{}/{}/{}".format(counter_best, counter_better, counter, len(users_to_recommend)))

            print("Test", folder, exam_folder)

            counter = 0
            counter_better = 0
            counter_best = 0
            users_to_recommend = np.arange(exam_train.n_users)[np.ediff1d(urm_exam_test_neg.indptr) > 0]
            for u in users_to_recommend:
                if inv_exam_user_mapper[u] in user_mapper.keys():
                    exam_user_in_dataset = user_mapper[inv_exam_user_mapper[u]]
                    counter += 1
                    if exam_profile_lengths[u] < profile_lengths[exam_user_in_dataset]:
                        counter_better += 1
                        if exam_profile_lengths[u] < 4:
                            counter_best += 1
                    #print(inv_exam_user_mapper[u], exam_profile_lengths[u], profile_lengths[exam_user_in_dataset])
            print("{}/{}/{}/{}".format(counter_best, counter_better, counter, len(users_to_recommend)))

        print()





