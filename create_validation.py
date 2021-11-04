import os
import numpy as np

from RecSysFramework.DataManager import Dataset
from RecSysFramework.DataManager.Splitter import LeaveKOut
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

from RecSysFramework.Utils import generate_URM_test_negative
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate


if __name__ == "__main__":

    splitter = EXPERIMENTAL_CONFIG['splitter']

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
        exam_folder_profile_lengths = np.ediff1d(exam_train.get_URM().indptr)
        exam_valid_users_to_evaluate = np.arange(urm_exam_valid_neg.shape[0])[np.ediff1d(urm_exam_valid_neg.indptr) > 0]
        exam_test_users_to_evaluate = np.arange(urm_exam_test_neg.shape[0])[np.ediff1d(urm_exam_test_neg.indptr) > 0]

        exam_urm = exam_train.get_URM() + exam_valid.get_URM()
        dataset = Dataset(
            exam_folder, base_folder=EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder,
            URM_dict={"URM_all": exam_urm},
            URM_mappers_dict={"URM_all": exam_train.get_URM_mapper()}
        )

        for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
            dataset_name = "{}-{}".format(exam_folder, fold)
            folder = EXPERIMENTAL_CONFIG['dataset_folder'] + dataset_name + os.sep
            train, test = splitter.split(dataset, random_seed=fold + 10)
            splitter.save_split((train, test), save_folder_path=folder)
            URM_test_negative = generate_URM_test_negative(
                train.get_URM(), test.get_URM(), random_seed=fold + 1, negative_samples=99, type="fixed"
            )
            save_compressed_csr_matrix(URM_test_negative, folder + "urm_neg.npz")
