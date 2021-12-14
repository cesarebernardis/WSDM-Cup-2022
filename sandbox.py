import os

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from utils import create_dataset_from_folder, compress_urm, stretch_urm


if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
        urm = exam_train.get_URM().tocsr()
        urm_valid = exam_valid.get_URM().tocsr()
        exam_profile_lengths = np.ediff1d(urm.indptr)
        vals, counts = np.unique(exam_profile_lengths, return_counts=True)
        #print(vals)
        #print(counts)
        #print()

        for folder in EXPERIMENTAL_CONFIG['datasets']:
          if exam_folder in folder:
            train, valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(folder)
            user_mapper, item_mapper = train.get_URM_mapper()
            urm = train.get_URM().tocsr()
            urm_valid = valid.get_URM().tocsr()
            new_urm_exam_valid_neg = stretch_urm(urm_exam_valid_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
            new_urm_exam_test_neg = stretch_urm(urm_exam_test_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)

            valid_in_train = 0
            train_in_neg = 0
            train_in_test_neg = 0
            counter = 0
            for u in range(urm.shape[0]):
                indices_in_valid = urm_valid.indices[urm_valid.indptr[u]:urm_valid.indptr[u+1]]
                indices_in_train = urm.indices[urm.indptr[u]:urm.indptr[u+1]]
                indices_in_neg = new_urm_exam_valid_neg.indices[new_urm_exam_valid_neg.indptr[u]:new_urm_exam_valid_neg.indptr[u+1]]
                indices_in_test_neg = new_urm_exam_test_neg.indices[new_urm_exam_test_neg.indptr[u]:new_urm_exam_test_neg.indptr[u+1]]
                #print(u, indices_in_valid, indices_in_train)
                if len(indices_in_neg) > 0:
                    if np.any(np.isin(indices_in_valid, indices_in_train)):
                        valid_in_train += 1
                    if np.any(np.isin(indices_in_train, indices_in_neg)):
                        train_in_neg += 1
                    if np.any(np.isin(indices_in_train, indices_in_test_neg)):
                        train_in_test_neg += 1
                    counter += 1

            print(exam_folder, folder)
            print("valid_in_train", valid_in_train)
            print("train_in_neg", train_in_neg)
            print("train_in_test_neg", train_in_test_neg)
            print("counter", counter)
            print()

        exam_profile_lengths = np.ediff1d((exam_train.get_URM() + exam_valid.get_URM()).indptr)
        vals, counts = np.unique(exam_profile_lengths, return_counts=True)
        #print(vals)
        #print(counts)
        #print()
    exit()


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





