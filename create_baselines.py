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


def compute_scores(folder, algorithm, urm, urm_negs, user_mapper, item_mapper, user_prefix="", item_prefix="",
                   is_test=False, save=True, exam_folder=None, compress=True):

    n_users, n_items = urm.shape
    inv_user_mapper = {v: k for k, v in user_mapper.items()}
    inv_item_mapper = {v: k for k, v in item_mapper.items()}

    if exam_folder is None:
        exam_folder = folder

    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + algorithm.RECOMMENDER_NAME + os.sep
    dataIO = DataIO(folder_path=output_folder_path)
    data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")

    recommender = algorithm(urm)
    recommender.fit(**data_dict["hyperparameters_best"])

    # if exam_folder != folder:
    #     _exam_train, _exam_valid, _, _ = create_dataset_from_folder(exam_folder)
    #     _exam_user_mapper, _exam_item_mapper = _exam_train.get_URM_mapper()
    #     _urm_seen = stretch_urm(_exam_train.get_URM(), _exam_user_mapper, _exam_item_mapper, user_mapper, item_mapper)
    #     if is_test:
    #         _urm_seen += stretch_urm(_exam_valid.get_URM(), _exam_user_mapper, _exam_item_mapper, user_mapper, item_mapper)
    #     recommender.set_URM_seen(_urm_seen)

    if not isinstance(urm_negs, list):
        urm_negs = [urm_negs]

    for i, urm_neg in enumerate(urm_negs):

        recfile = output_folder_path
        if is_test:
            recfile += exam_folder + "_test_scores"
        else:
            recfile += exam_folder + "_valid_scores"

        if len(urm_negs) > 1:
            recfile += "_{}".format(i)

        recfile += ".tsv"

        if compress:
            if not recfile.endswith(".gz"):
                recfile += ".gz"
            file = gzip.open(recfile, "wt")
        else:
            file = open(recfile, "w")

        if save:
            print("userId\titemId\tscore", file=file)

        batchsize = 400
        users_to_recommend = np.arange(n_users)[np.ediff1d(urm_neg.indptr) > 0]

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
                        minimum = min(sc[sc != -np.inf])
                        minimum -= abs(minimum) * 0.02
                    else:
                        minimum = 1e-8
                    sc = np.maximum(sc, minimum)
                    for i_idx, i in enumerate(items_to_compute[u_idx].tolist()):
                        print("{}\t{}\t{}".format(user_prefix + inv_user_mapper[u].strip(),
                                                  item_prefix + inv_item_mapper[i].strip(),
                                                  sc[i_idx]), file=file)

        file.close()




if __name__ == "__main__":

    max_cutoff = max(EXPERIMENTAL_CONFIG['cutoffs'])
    splitter = EXPERIMENTAL_CONFIG['splitter']

    for i, folder in enumerate(EXPERIMENTAL_CONFIG['datasets']):

        URM_seen = None

        train, valid, urm_valid_neg, urm_test_neg = create_dataset_from_folder(folder)
        user_mapper, item_mapper = train.get_URM_mapper()

        if "kcore" in folder:
            kcore = 5
        else:
            kcore = 0

        for algorithm in EXPERIMENTAL_CONFIG['baselines']:

            output_folder_path = train.get_complete_folder() + os.sep + algorithm.RECOMMENDER_NAME + os.sep
            to_recompute = False

            try:
                filename = algorithm.RECOMMENDER_NAME + "_metadata"
                dataio = DataIO(output_folder_path)
                hpo = dataio.load_data(file_name=filename)

                for i, r in enumerate(hpo['result_on_validation_list']):
                    if r is None:
                        hpo['hyperparameters_list'][i] = None
                        to_recompute = True

                dataio.save_data(data_dict_to_save=hpo, file_name=filename)
            except FileNotFoundError:
                pass

            run_parameter_search(
                algorithm, "leave_one_out", train, valid, output_folder_path=output_folder_path,
                metric_to_optimize="NDCG", cutoff_to_optimize=10, resume_from_saved=True,
                train_user_kcore=kcore, train_item_kcore=kcore, URM_seen=URM_seen,
                n_cases=50, n_random_starts=30, save_model="no", URM_validation_negatives=urm_valid_neg
            )

            recfile = output_folder_path + os.sep + folder + "_valid_scores.tsv.gz"
            if not os.path.isfile(recfile):
                compute_scores(folder, algorithm, train.get_URM(), urm_valid_neg,
                        user_mapper=user_mapper, item_mapper=item_mapper, is_test=False)

            for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:

                if exam_folder in folder:

                    exam_train, exam_valid, exam_urm_valid_neg, exam_urm_test_neg = create_dataset_from_folder(exam_folder)
                    exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
                    exam_valid_urm = exam_valid.get_URM()

                    exam_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep

                    for fold in range(EXPERIMENTAL_CONFIG['n_folds'])[:0]:

                        validation_folder = exam_folder + "-" + str(fold)
                        #recfile = output_folder_path + os.sep + validation_folder + "_valid_scores.tsv.gz"
                        #if not os.path.isfile(recfile) or to_recompute:
                        matrices_folder = exam_folder_path + validation_folder + os.sep
                        extra_exam_urm_valid_neg = [
                            stretch_urm(load_compressed_csr_matrix(matrices_folder + "urm_neg_{}.npz".format(f)),
                                        exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                            for f in range(EXPERIMENTAL_CONFIG['n_folds'])
                        ]
                        extra_train, extra_test = splitter.load_split(matrices_folder)
                        compute_scores(folder, algorithm, extra_train.get_URM(), extra_exam_urm_valid_neg,
                                       user_mapper=user_mapper, item_mapper=item_mapper,
                                       is_test=False, exam_folder=validation_folder)

                    exam_urm_valid_neg = stretch_urm(exam_urm_valid_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)

                    recfile = output_folder_path + os.sep + exam_folder + "_valid_scores.tsv.gz"
                    if not os.path.isfile(recfile) or to_recompute:
                        compute_scores(folder, algorithm, train.get_URM(), exam_urm_valid_neg,
                                       user_mapper=user_mapper, item_mapper=item_mapper,
                                       is_test=False, exam_folder=exam_folder)

                    if any(not os.path.isfile(output_folder_path + os.sep + "{}_valid_scores_{}.tsv.gz".format(exam_folder, f))
                           for f in range(EXPERIMENTAL_CONFIG['n_folds'])):
                        exam_urm_valid_neg = [
                            stretch_urm(load_compressed_csr_matrix(exam_folder_path + "urm_valid_neg_{}.npz".format(f)),
                                        exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                            for f in range(EXPERIMENTAL_CONFIG['n_folds'])
                        ]
                        compute_scores(folder, algorithm, train.get_URM(), exam_urm_valid_neg,
                                       user_mapper=user_mapper, item_mapper=item_mapper,
                                       is_test=False, exam_folder=exam_folder)

                    if exam_urm_test_neg is not None:
                        recfile = output_folder_path + os.sep + exam_folder + "_test_scores.tsv.gz"
                        if not os.path.isfile(recfile) or to_recompute:
                            exam_urm_test_neg = stretch_urm(exam_urm_test_neg, exam_user_mapper, exam_item_mapper, user_mapper, item_mapper)
                            urm = train.get_URM()
                            urm_valid = valid.get_URM()
                            urm_valid.data[:] = max(urm.data)
                            urm += urm_valid
                            compute_scores(folder, algorithm, urm, exam_urm_test_neg,
                                           user_mapper=user_mapper, item_mapper=item_mapper,
                                           is_test=True, exam_folder=exam_folder)

