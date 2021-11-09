import os
import gzip

import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.ParameterTuning.Utils import run_parameter_search
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from utils import create_dataset_from_folder, stretch_urm


def compute_scores(folder, algorithm, urm, urm_neg, user_mapper, item_mapper, user_prefix="", item_prefix="",
                   is_test=False, save=True, exam_folder=None, compress=True):

    n_users, n_items = urm.shape
    inv_user_mapper = {v: k for k, v in user_mapper.items()}
    inv_item_mapper = {v: k for k, v in item_mapper.items()}

    if exam_folder is None:
        exam_folder = folder

    users_to_recommend = np.arange(n_users)[np.ediff1d(urm_neg.indptr) > 0]

    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + algorithm.RECOMMENDER_NAME + os.sep
    dataIO = DataIO(folder_path=output_folder_path)
    data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")

    recommender = algorithm(urm)
    recommender.fit(**data_dict["hyperparameters_best"])

    recfile = output_folder_path
    if is_test:
        recfile += exam_folder + "_test_scores.tsv"
    else:
        recfile += exam_folder + "_valid_scores.tsv"

    if compress:
        if not recfile.endswith(".gz"):
            recfile += ".gz"
        file = gzip.open(recfile, "wt")
    else:
        file = open(recfile, "w")

    if save:
        print("userId\titemId\tscore", file=file)

    batchsize = 400

    for start in range(0, len(users_to_recommend) + 1, batchsize):

        end = min(start + batchsize, len(users_to_recommend))

        items_to_compute = []
        for u in users_to_recommend[start:end]:
            start_pos = urm_neg.indptr[u]
            end_pos = urm_neg.indptr[u + 1]
            items_to_compute.append(urm_neg.indices[start_pos:end_pos])

        _, scores = recommender.recommend(
            users_to_recommend[start:end], cutoff=100, remove_seen_flag=True,
            items_to_compute=items_to_compute, return_scores=True, filter_at_recommendation=False
        )

        if save:
            for u_idx, u in enumerate(users_to_recommend[start:end]):
                sc = scores[u_idx, :][items_to_compute[u_idx]]
                minimum = min(sc)
                maximum = max(sc)
                if minimum != maximum:
                    minimum = min(sc[sc != -np.inf]) - 0.1
                else:
                    minimum = 1e-8
                for i_idx, i in enumerate(items_to_compute[u_idx].tolist()):
                    s = max(sc[i_idx], minimum)
                    print("{}\t{}\t{}".format(user_prefix + inv_user_mapper[u].strip(),
                                              item_prefix + inv_item_mapper[i].strip(),
                                              s), file=file)

    file.close()




if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])
    splitter = EXPERIMENTAL_CONFIG['splitter']

    for i, folder in enumerate(EXPERIMENTAL_CONFIG['datasets']):

        train, valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(folder)
        user_mapper, item_mapper = train.get_URM_mapper()

        for algorithm in EXPERIMENTAL_CONFIG['baselines']:

            output_folder_path = train.get_complete_folder() + os.sep + algorithm.RECOMMENDER_NAME + os.sep

            run_parameter_search(
                algorithm, "leave_one_out", train, valid, output_folder_path=output_folder_path,
                metric_to_optimize="NDCG", cutoff_to_optimize=10, resume_from_saved=True,
                n_cases=50, n_random_starts=35, save_model="no", URM_validation_negatives=urm_valid_neg
            )

            recfile = output_folder_path + os.sep + folder + "_valid_scores.tsv.gz"
            if not os.path.isfile(recfile):
                compute_scores(folder, algorithm, train.get_URM(), urm_valid_neg,
                        user_mapper=user_mapper, item_mapper=item_mapper, user_prefix=folder, is_test=False)

            for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

                if exam_folder in folder:

                    for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                        validation_folder = exam_folder + "-" + str(fold)
                        recfile = output_folder_path + os.sep + validation_folder + "_valid_scores.tsv.gz"
                        if not os.path.isfile(recfile):
                            matrices_folder = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + validation_folder + os.sep
                            with open("{}URM_all_train_mapper".format(matrices_folder), "rb") as file:
                                exam_user_mapper, exam_item_mapper = pkl.load(file)
                            exam_urm_valid_neg = load_compressed_csr_matrix(matrices_folder + "urm_neg.npz")
                            exam_urm_valid_neg = stretch_urm(exam_urm_valid_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                            urm = train.get_URM() - 100 * exam_urm_valid_neg.astype(np.float32)
                            urm.data[urm.data < 0.] = 0.
                            urm.eliminate_zeros()
                            compute_scores(folder, algorithm, urm, exam_urm_valid_neg,
                                           user_mapper=user_mapper, item_mapper=item_mapper,
                                           user_prefix=exam_folder, is_test=False, exam_folder=validation_folder)

                    exam_train, exam_valid, exam_urm_valid_neg, exam_urm_test_neg = create_dataset_from_folder(exam_folder)
                    exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
                    exam_valid_urm = exam_valid.get_URM()

                    if folder != exam_folder:
                        exam_valid_urm = stretch_urm(exam_valid_urm, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                        exam_urm_valid_neg = stretch_urm(exam_urm_valid_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                        exam_urm_test_neg = stretch_urm(exam_urm_test_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)

                    recfile = output_folder_path + os.sep + exam_folder + "_valid_scores.tsv.gz"
                    if not os.path.isfile(recfile):
                        compute_scores(folder, algorithm, train.get_URM(), exam_urm_valid_neg,
                                       user_mapper=user_mapper, item_mapper=item_mapper,
                                       user_prefix=exam_folder, is_test=False, exam_folder=exam_folder)

                    if exam_urm_test_neg is not None:
                        recfile = output_folder_path + os.sep + exam_folder + "_test_scores.tsv.gz"
                        if not os.path.isfile(recfile):
                            urm = train.get_URM() + valid.get_URM()
                            if exam_folder != folder:
                                urm += exam_valid_urm
                            compute_scores(folder, algorithm, urm, exam_urm_test_neg,
                                           user_mapper=user_mapper, item_mapper=item_mapper,
                                           user_prefix=exam_folder, is_test=True, exam_folder=exam_folder)

