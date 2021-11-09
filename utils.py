import os
import gzip
import shutil
import zipfile
import numpy as np
import scipy.sparse as sps
import pickle as pkl

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


def create_dataset_from_folder(folder):

    basepath = EXPERIMENTAL_CONFIG["dataset_folder"] + folder + os.sep

    urm = sps.load_npz(basepath + "urm.npz")
    urm_valid = sps.load_npz(basepath + "urm_valid.npz")
    urm_valid_neg = sps.load_npz(basepath + "urm_valid_neg.npz")

    if os.path.isfile(basepath + "urm_test_neg.npz"):
        urm_test_neg = sps.load_npz(basepath + "urm_test_neg.npz").tocsr()
    else:
        urm_test_neg = None

    with open(basepath + "user_mapping.pkl", "rb") as handle:
        user_mapping = pkl.load(handle)

    with open(basepath + "item_mapping.pkl", "rb") as handle:
        item_mapping = pkl.load(handle)

    train = Dataset(
        dataset_name=folder, base_folder=basepath, URM_dict={"URM_all": urm},
        URM_mappers_dict={"URM_all": (user_mapping, item_mapping)}
    )

    valid = Dataset(
        dataset_name=folder, base_folder=basepath, URM_dict={"URM_all": urm_valid},
        URM_mappers_dict={"URM_all": (user_mapping, item_mapping)}
    )

    return train, valid, urm_valid_neg.tocsr(), urm_test_neg


def stretch_urm(a, user_mapper_a, item_mapper_a, user_mapper_b, item_mapper_b):

    inv_user_mapper_a = {v: k for k, v in user_mapper_a.items()}
    inv_item_mapper_a = {v: k for k, v in item_mapper_a.items()}

    inv_user_mapper_b = {v: k for k, v in user_mapper_b.items()}
    inv_item_mapper_b = {v: k for k, v in item_mapper_b.items()}

    rows = []
    cols = []
    data = []

    user_mapper = np.zeros(a.shape[0], dtype=np.int32)
    for k in inv_user_mapper_a.keys():
        user_mapper[k] = user_mapper_b[inv_user_mapper_a[k]]

    item_mapper = np.zeros(a.shape[1], dtype=np.int32)
    for k in inv_item_mapper_a.keys():
        item_mapper[k] = item_mapper_b[inv_item_mapper_a[k]]

    for user in range(len(user_mapper)):
        start = a.indptr[user]
        end = a.indptr[user + 1]
        if start != end:
            rows.append(np.ones(end - start, dtype=np.int32) * user_mapper[user])
            cols.append(item_mapper[a.indices[start:end]])
            data.append(a.data[start:end])

    return sps.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(user_mapper_b), len(item_mapper_b)))


def compress_urm(a, user_mapper_a, item_mapper_a, user_mapper_b, item_mapper_b):

    inv_user_mapper_a = {v: k for k, v in user_mapper_a.items()}
    inv_item_mapper_a = {v: k for k, v in item_mapper_a.items()}

    inv_user_mapper_b = {v: k for k, v in user_mapper_b.items()}
    inv_item_mapper_b = {v: k for k, v in item_mapper_b.items()}

    new_urm = a.copy()

    user_mapper = - np.ones(a.shape[0], dtype=np.int32)
    for k in inv_user_mapper_b.keys():
        user_mapper[user_mapper_a[inv_user_mapper_b[k]]] = k
    mask = user_mapper >= 0
    new_urm = new_urm[mask, :]
    new_urm = new_urm[user_mapper[mask], :]

    item_mapper = - np.ones(a.shape[1], dtype=np.int32)
    for k in inv_item_mapper_b.keys():
        item_mapper[item_mapper_a[inv_item_mapper_b[k]]] = k
    mask = item_mapper >= 0
    new_urm = new_urm[:, mask]
    new_urm = new_urm[:, item_mapper[mask]]

    return new_urm.tocsr()



def row_minmax_scaling(urm):
    littleval = 1e-08
    for u in range(urm.shape[0]):
        start = urm.indptr[u]
        end = urm.indptr[u+1]
        if start != end:
            d = urm.data[start:end]
            minimum = min(d)
            maximum = max(d)
            if minimum == maximum:
                urm.data[start:end] = littleval
            else:
                urm.data[start:end] = (d - minimum) / (maximum - minimum) + littleval
    return urm


def read_ratings(filename, user_mapping, item_mapping):

    urm_row = []
    urm_col = []
    urm_data = []

    n_users = len(user_mapping)
    n_items = len(item_mapping)

    compress = False
    if filename.endswith(".gz"):
        compress = True
    elif os.path.isfile(filename + ".gz"):
        filename += ".gz"
        compress = True

    if compress:
        file = gzip.open(filename, "rt")
    else:
        file = open(filename, "r")

    jump_header = True
    for line in file:

        if jump_header:
            jump_header = False
            continue

        split_row = line.strip().split("\t")
        uid = user_mapping[split_row[0][2:].strip()]
        iid = item_mapping[split_row[1].strip()]

        urm_row.append(uid)
        urm_col.append(iid)
        urm_data.append(float(split_row[2].strip()))
        
    file.close()

    return sps.csr_matrix((urm_data, (urm_row, urm_col)), shape=(n_users, n_items), dtype=np.float32)


def output_scores(filename, urm, user_mapper, item_mapper, user_prefix="", item_prefix="", compress=True):

    inv_user_mapper = {v: k for k, v in user_mapper.items()}
    inv_item_mapper = {v: k for k, v in item_mapper.items()}

    if compress:
        if not filename.endswith(".gz"):
            filename += ".gz"
        file = gzip.open(filename, "wt")
    else:
        file = open(filename, "w")

    print("userId\titemId\tscore", file=file)

    for u in range(urm.shape[0]):
        if urm.indptr[u] != urm.indptr[u+1]:
            for i_idx in range(urm.indptr[u+1] - urm.indptr[u]):
                i = urm.indices[urm.indptr[u] + i_idx]
                score = urm.data[urm.indptr[u] + i_idx]
                print("{}\t{}\t{:.12f}".format(
                    user_prefix + inv_user_mapper[u],
                    item_prefix + inv_item_mapper[i], score), file=file)

    file.close()


def evaluate(ratings, exam_test_urm, cutoff=10):
    scores = np.zeros(exam_test_urm.shape[0], dtype=np.float32)
    n_evals = 0
    counter = 0
    for u in range(exam_test_urm.shape[0]):
        if exam_test_urm.indptr[u] != exam_test_urm.indptr[u+1]:
            ranking = np.argsort(ratings.data[ratings.indptr[u]:ratings.indptr[u+1]])[::-1]
            #if ratings.indptr[u] != ratings.indptr[u+1]:
            #    if not np.any(np.isin(ratings.indices[ratings.indptr[u]:ratings.indptr[u+1]],
            #                exam_test_urm.indices[exam_test_urm.indptr[u]:exam_test_urm.indptr[u+1]])):
            #        counter += 1
            scores[u] = ndcg(ratings.indices[ratings.indptr[u]:ratings.indptr[u+1]][ranking],
                             exam_test_urm.indices[exam_test_urm.indptr[u]:exam_test_urm.indptr[u+1]], cutoff)
            n_evals += 1
    #print("counter", counter, n_evals)
    return scores, n_evals


def make_submission():

    sub_filename = "submission" + os.sep + "submission.zip"

    os.makedirs("submission" + os.sep + "t1", exist_ok=True)
    os.makedirs("submission" + os.sep + "t2", exist_ok=True)

    if os.path.exists(sub_filename):
        os.remove(sub_filename)

    for folder in ["t1", "t2"]:
        for target in ["valid", "test"]:
            shutil.copyfile(os.sep.join(["datasets", folder, target + "_scores.tsv"]),
                            os.sep.join(["submission", folder, target + "_pred.tsv"]))

    with zipfile.ZipFile(sub_filename, mode='a') as zfile:
        for folder in ["t1", "t2"]:
            for target in ["valid", "test"]:
                p = ["submission", folder, target + "_pred.tsv"]
                zfile.write(os.sep.join(p), arcname=os.sep.join(p[1:]))


def first_level_ensemble(folder, exam_folder, exam_valid, exam_folder_profile_lengths, user_masks):

    exam_user_mapper, exam_item_mapper = exam_valid.get_URM_mapper()

    with open(EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + "best-ensemble-weights.pkl", "rb") as file:
        ensemble_weights = pkl.load(file)

    urm_valid_total = None
    urm_test_total = None

    for recname in ensemble_weights['cold'].keys():

        output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
        try:
            ratings = read_ratings(output_folder_path + exam_folder + "_valid_scores.tsv.gz",
                                   exam_user_mapper, exam_item_mapper)
        except Exception as e:
            continue

        algo_weights = np.zeros(ratings.shape[0], dtype=np.float32)
        for usertype in ["cold", "quite-cold", "quite-warm", "warm"]:
            algo_weights[user_masks[usertype]] = ensemble_weights[usertype][recname] * \
                                                 ensemble_weights[usertype][recname + "-sign"]

        weights = sps.diags(algo_weights, dtype=np.float32)
        ratings = weights.dot(row_minmax_scaling(ratings))
        user_ndcg, n_evals = evaluate(ratings, exam_valid.get_URM(), cutoff=10)
        avg_ndcg = np.sum(user_ndcg) / n_evals
        outstr = "\t".join(map(str, ["Validation", exam_folder, folder, recname, avg_ndcg]))
        print(outstr)

        if urm_valid_total is None:
            urm_valid_total = ratings
        else:
            urm_valid_total += ratings

        ratings = read_ratings(output_folder_path + exam_folder + "_test_scores.tsv.gz",
                               exam_user_mapper, exam_item_mapper)
        ratings = weights.dot(row_minmax_scaling(ratings))

        if urm_test_total is None:
            urm_test_total = ratings
        else:
            urm_test_total += ratings

    user_ndcg, n_evals = evaluate(urm_valid_total, exam_valid.get_URM(), cutoff=10)
    avg_ndcg = np.sum(user_ndcg) / n_evals
    outstr = "\t".join(map(str, ["Validation", exam_folder, folder, "First level Ensemble", avg_ndcg]))
    print(outstr)

    return urm_valid_total, urm_test_total

