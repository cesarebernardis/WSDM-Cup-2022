import os
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from utils import create_dataset_from_folder, compress_urm


def print_info(_folder):
    print("--- {} ---".format(_folder))
    train, valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(_folder)
    urm = train.get_URM()
    print("urm", urm.shape, len(urm.data))
    urm = valid.get_URM()
    print("urm_valid", urm.shape, len(urm.data))
    print("urm_valid_neg", urm_valid_neg.shape, len(urm_valid_neg.data))
    if urm_test_neg is not None:
        print("urm_test_neg", urm_test_neg.shape, len(urm_test_neg.data))


for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

    exam_train, exam_valid, exam_urm_valid_neg, exam_urm_test_neg = create_dataset_from_folder(exam_folder)
    exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
    print_info(exam_folder)

    for folder in EXPERIMENTAL_CONFIG['datasets']:

        if folder != exam_folder and exam_folder in folder:

            folder_train, folder_valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(folder)
            user_mapper, item_mapper = folder_train.get_URM_mapper()

            urm = compress_urm(folder_train.get_URM(), user_mapper, item_mapper, exam_user_mapper, exam_item_mapper, skip_users=True)
            urm_valid = compress_urm(folder_valid.get_URM(), user_mapper, item_mapper, exam_user_mapper, exam_item_mapper, skip_users=True)
            urm_valid_neg = compress_urm(urm_valid_neg, user_mapper, item_mapper, exam_user_mapper, exam_item_mapper, skip_users=True)

            add_row = []
            add_col = []
            for u in range(urm_valid.shape[0]):
                if urm_valid.indptr[u + 1] != urm_valid.indptr[u]:
                    train_indices = urm.indices[urm.indptr[u]:urm.indptr[u+1]]
                    valid_neg_indices = urm_valid_neg.indices[urm_valid_neg.indptr[u]:urm_valid_neg.indptr[u+1]]
                    valid_indices = urm_valid.indices[urm_valid.indptr[u]:urm_valid.indptr[u+1]]
                    if not np.any(np.isin(valid_indices, train_indices)) and np.all(np.isin(valid_indices, valid_neg_indices)):
                        missing = 100 - (urm_valid_neg.indptr[u + 1] - urm_valid_neg.indptr[u])
                        if missing > 0:
                            p = np.ones(urm_valid.shape[1])
                            p[train_indices] = 0.
                            p[valid_neg_indices] = 0.
                            p[valid_indices] = 0
                            p = p / sum(p)
                            add_row.append(np.repeat(u, missing))
                            add_col.append(np.random.choice(len(p), missing, replace=False, p=p))
                            #urm_valid_neg.data[urm_valid_neg.indptr[u]:urm_valid_neg.indptr[u+1]] = 0.
                            #urm_valid.data[urm_valid.indptr[u]:urm_valid.indptr[u+1]] = 0.
                    else:
                        urm_valid.data[urm_valid.indptr[u]:urm_valid.indptr[u + 1]] = 0.
                        urm_valid_neg.data[urm_valid_neg.indptr[u]:urm_valid_neg.indptr[u + 1]] = 0.
                elif urm_valid_neg.indptr[u + 1] != urm_valid_neg.indptr[u]:
                    urm_valid_neg.data[urm_valid_neg.indptr[u]:urm_valid_neg.indptr[u + 1]] = 0.

            urm_valid.eliminate_zeros()
            urm_valid_neg.eliminate_zeros()

            add_row = np.concatenate(add_row)
            add_col = np.concatenate(add_col)
            urm_valid_neg += sps.csr_matrix((np.ones(len(add_row)), (add_row, add_col)), shape=urm_valid_neg.shape)

            if urm_test_neg is not None:
                urm_test_neg = compress_urm(urm_test_neg, user_mapper, item_mapper, exam_user_mapper, exam_item_mapper, skip_users=True)

            folder_path = EXPERIMENTAL_CONFIG["dataset_folder"] + folder + "-onlyt" + os.sep
            os.makedirs(folder_path, exist_ok=True)
            sps.save_npz(folder_path + "urm.npz", urm, compressed=True)
            sps.save_npz(folder_path + "urm_valid.npz", urm_valid, compressed=True)
            sps.save_npz(folder_path + "urm_valid_neg.npz", urm_valid_neg, compressed=True)
            if urm_test_neg is not None:
                sps.save_npz(folder_path + "urm_test_neg.npz", urm_test_neg, compressed=True)

            with open(folder_path + "user_mapping.pkl", "wb") as handle:
                pkl.dump(user_mapper, handle, protocol=pkl.HIGHEST_PROTOCOL)

            with open(folder_path + "item_mapping.pkl", "wb") as handle:
                pkl.dump(exam_item_mapper, handle, protocol=pkl.HIGHEST_PROTOCOL)

            print_info(folder + "-onlyt")


