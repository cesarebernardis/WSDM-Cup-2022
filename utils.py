import os
import gzip
import shutil
import zipfile
import numpy as np
import scipy.sparse as sps
import pickle as pkl
import pandas as pd
import sklearn

from RecSysFramework.Recommender.MatrixFactorization import IALS
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
        if inv_user_mapper_a[k] in user_mapper_b:
            user_mapper[k] = user_mapper_b[inv_user_mapper_a[k]]

    item_mapper = np.zeros(a.shape[1], dtype=np.int32)
    for k in inv_item_mapper_a.keys():
        if inv_item_mapper_a[k] in item_mapper_b:
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


def compress_urm(a, user_mapper_a, item_mapper_a, user_mapper_b, item_mapper_b, skip_users=False, skip_items=False):
    inv_user_mapper_a = {v: k for k, v in user_mapper_a.items()}
    inv_item_mapper_a = {v: k for k, v in item_mapper_a.items()}

    inv_user_mapper_b = {v: k for k, v in user_mapper_b.items()}
    inv_item_mapper_b = {v: k for k, v in item_mapper_b.items()}

    new_urm = a.copy()

    if not skip_users:
        user_mapper = - np.ones(len(user_mapper_b), dtype=np.int32)
        for k in inv_user_mapper_b.keys():
            if inv_user_mapper_b[k] in user_mapper_a.keys():
                user_mapper[k] = user_mapper_a[inv_user_mapper_b[k]]
        mask = user_mapper >= 0
        # new_urm = new_urm[mask, :]
        new_urm = new_urm[user_mapper[mask], :]

    if not skip_items:
        item_mapper = - np.ones(len(item_mapper_b), dtype=np.int32)
        for k in inv_item_mapper_b.keys():
            if inv_item_mapper_b[k] in item_mapper_a.keys():
                item_mapper[k] = item_mapper_a[inv_item_mapper_b[k]]
        mask = item_mapper >= 0
        # new_urm = new_urm[:, mask]
        new_urm = new_urm[:, item_mapper[mask]]

    new_urm.eliminate_zeros()

    return new_urm.tocsr()


def row_minmax_scaling(urm):
    littleval = 1e-08
    for u in range(urm.shape[0]):
        start = urm.indptr[u]
        end = urm.indptr[u + 1]
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
        if urm.indptr[u] != urm.indptr[u + 1]:
            for i_idx in range(urm.indptr[u + 1] - urm.indptr[u]):
                i = urm.indices[urm.indptr[u] + i_idx]
                score = urm.data[urm.indptr[u] + i_idx]
                print("{}\t{}\t{:.12f}".format(
                    user_prefix + inv_user_mapper[u],
                    item_prefix + inv_item_mapper[i], score), file=file)

    file.close()


def evaluate(ratings, exam_test_urm, cheat=False, cutoff=10):
    scores = np.zeros(exam_test_urm.shape[0], dtype=np.float32)
    n_evals = 0
    counter = 0
    jumped = 0
    if cheat:
        print("WARNING! CHEATING!")
    for u in range(exam_test_urm.shape[0]):
        if exam_test_urm.indptr[u] != exam_test_urm.indptr[u + 1]:
            if cheat and ratings.indptr[u] == ratings.indptr[u + 1]:
                jumped += 1
                continue
            ranking = np.argsort(ratings.data[ratings.indptr[u]:ratings.indptr[u + 1]])[::-1]
            scores[u] = ndcg(ratings.indices[ratings.indptr[u]:ratings.indptr[u + 1]][ranking],
                             exam_test_urm.indices[exam_test_urm.indptr[u]:exam_test_urm.indptr[u + 1]], cutoff)
            n_evals += 1
    if cheat:
        print("Jumped {} evaluations".format(jumped))
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

    with open(EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + "best-ensemble-weights-{}.pkl".format(
            exam_folder), "rb") as file:
        ensemble_weights = pkl.load(file)

    urm_valid_total = None
    urm_test_total = None

    # exam_train, _, _, _ = create_dataset_from_folder(exam_folder)
    # pop = np.ediff1d(exam_train.get_URM().tocsc().indptr)
    # itempop = np.power(pop + 1, 0.02)
    # itempop = np.zeros(len(exam_item_mapper))
    # itempop[np.argsort(pop)[-1000:]] = 1.

    # print(itempop, min(itempop), max(itempop))
    # itempop = sps.diags(itempop)

    for recname in ensemble_weights['cold'].keys():

        output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
        try:
            ratings = read_ratings(output_folder_path + exam_folder + "_valid_scores.tsv.gz",
                                   exam_user_mapper, exam_item_mapper)
        except Exception as e:
            continue

        """
        user_ndcg, n_evals = evaluate(ratings, exam_valid.get_URM(), cutoff=10)
        avg_ndcg = np.sum(user_ndcg) / n_evals
        outstr = "\t".join(map(str, ["Validation", exam_folder, folder, recname, avg_ndcg]))
        print(outstr)

        user_ndcg, n_evals = evaluate(row_minmax_scaling(ratings).dot(itempop), exam_valid.get_URM(), cutoff=10)
        avg_ndcg = np.sum(user_ndcg) / n_evals
        outstr = "\t".join(map(str, ["Validation", exam_folder, folder, recname, avg_ndcg]))
        print("POP BOOSTED", outstr)
        continue
        """

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

    # user_ndcg, n_evals = evaluate(urm_valid_total, exam_valid.get_URM(), cutoff=10)
    # avg_ndcg = np.sum(user_ndcg) / n_evals
    # outstr = "\t".join(map(str, ["Validation", exam_folder, folder, "First level Ensemble", avg_ndcg]))
    # print(outstr)

    return urm_valid_total, urm_test_total


class FeatureGenerator:

    def __init__(self, exam_folder, n_folds=0):
        if n_folds is None:
            self.n_folds = EXPERIMENTAL_CONFIG['n_folds']
        else:
            self.n_folds = n_folds
        self.exam_folder = exam_folder
        self.exam_train, self.exam_valid, self.urm_exam_valid_neg, self.urm_exam_test_neg = create_dataset_from_folder(
            exam_folder)
        exam_user_mapper, exam_item_mapper = self.exam_train.get_URM_mapper()

        self.validations = []
        self.validation_negs = []
        self.exam_profile_lengths = []
        self.exam_itempops = []
        self.user_mappers = []
        self.item_mappers = []
        self.urms = []
        for fold in range(self.n_folds):
            validation_path = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep + exam_folder + "-" + str(
                fold) + os.sep
            with open(validation_path + "URM_all_train_mapper", "rb") as file:
                um, im = pkl.load(file)
                self.user_mappers.append(um)
                self.item_mappers.append(im)
            urm_train = sps.load_npz(validation_path + "URM_all_train.npz")
            self.urms.append(urm_train)
            self.exam_itempops.append(np.ediff1d(urm_train.tocsc().indptr))
            self.validations.append(sps.load_npz(validation_path + "URM_all_test.npz").tocoo())
            self.validation_negs.append(load_compressed_csr_matrix(validation_path + "urm_neg.npz").tocoo())

        self.user_mappers.append(exam_user_mapper)
        self.item_mappers.append(exam_item_mapper)
        self.validation_negs.append(self.urm_exam_valid_neg.tocoo())
        self.validations.append(self.exam_valid.get_URM().tocoo())
        self.urms.append(self.exam_train.get_URM())
        self.exam_profile_lengths.append(np.ediff1d(self.urms[-1].indptr))
        self.exam_itempops.append(np.ediff1d(self.urms[-1].tocsc().indptr))

        self.user_mappers.append(exam_user_mapper)
        self.item_mappers.append(exam_item_mapper)
        self.validation_negs.append(self.urm_exam_test_neg.tocoo())
        valid_urm = self.exam_valid.get_URM()
        valid_urm.data[:] = 4.5
        self.urms.append(self.exam_train.get_URM() + valid_urm)
        self.exam_profile_lengths.append(np.ediff1d(self.urms[-1].indptr))
        self.exam_itempops.append(np.ediff1d(self.urms[-1].tocsc().indptr))

        self.basic_dfs = []
        for fold in range(self.n_folds + 2):
            self.basic_dfs.append(pd.DataFrame({'user': self.validation_negs[fold].row.copy(),
                                                'item': self.validation_negs[fold].col.copy()}))
            fold_fpl = np.ediff1d(self.urms[fold].tocsr().indptr)
            fold_itempop = np.ediff1d(self.urms[fold].tocsc().indptr)
            pl_df = pd.DataFrame({'user': np.arange(len(fold_fpl)), 'pl': fold_fpl, 'pl_log': np.log(fold_fpl + 1)})
            pop_df = pd.DataFrame({'item': np.arange(len(fold_itempop)), 'popularity': fold_itempop,
                                   'popularity_log': np.log(fold_itempop + 1)})
            self.basic_dfs[fold] = self.basic_dfs[fold].merge(pop_df, on=["item"], how="left", sort=True)
            self.basic_dfs[fold] = self.basic_dfs[fold].merge(pl_df, on=["user"], how="left", sort=True)

    def _get_mapper_converters(self, user_mapper, item_mapper):

        user_converters = []
        for i in range(len(self.user_mappers)):
            converter = np.zeros(len(self.user_mappers[i]), dtype=np.int32)
            inv_mapper = {v: k for k, v in self.user_mappers[i].items()}
            for k, v in inv_mapper.items():
                converter[k] = user_mapper[v]
            user_converters.append(converter)
            assert len(np.unique(converter)) == len(inv_mapper), "Errore user: {} - {}".format(len(np.unique(converter)), len(inv_mapper))

        item_converters = []
        for i in range(len(self.item_mappers)):
            converter = np.zeros(len(self.item_mappers[i]), dtype=np.int32)
            inv_mapper = {v: k for k, v in self.item_mappers[i].items()}
            for k, v in inv_mapper.items():
                converter[k] = item_mapper[v]
            item_converters.append(converter)
            assert len(np.unique(converter)) == len(inv_mapper), "Errore item: {} - {}".format(len(np.unique(converter)), len(inv_mapper))

        return user_converters, item_converters


    def _prepare_folder_urms(self, folder):
        folder_train, folder_valid, _, _ = create_dataset_from_folder(folder)
        user_mapper, item_mapper = folder_train.get_URM_mapper()
        folder_urm = folder_train.get_URM()
        folder_valid_urm = folder_valid.get_URM()
        folder_valid_urm.data[:] = 4.5
        folder_test_urm = folder_urm + folder_valid_urm
        return folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper


    def load_folder_features(self, folder, add_dataset_features_to_last_level_df=True):

        folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper = self._prepare_folder_urms(folder)

        fpl = np.ediff1d(folder_urm.tocsr().indptr)
        itempop = np.ediff1d(folder_urm.tocsc().indptr)

        test_fpl = np.ediff1d(folder_test_urm.tocsr().indptr)
        test_itempop = np.ediff1d(folder_test_urm.tocsc().indptr)

        ratings = [pd.DataFrame({'user': self.validation_negs[fold].row.copy(),
                                 'item': self.validation_negs[fold].col.copy()})
                   for fold in range(len(self.validation_negs))]

        user_converters, item_converters = self._get_mapper_converters(user_mapper, item_mapper)

        # for i, epl in enumerate(self.exam_profile_lengths):
        #     if i == len(self.exam_profile_lengths) - 1:
        #         pl_diff = test_fpl[user_converters[i]] - epl
        #     else:
        #         pl_diff = fpl[user_converters[i]] - epl
        #     pl_diff = np.maximum(pl_diff, 0)
        #     local_df = pd.DataFrame(
        #         {'user': np.arange(len(pl_diff)), 'pl_diff': pl_diff, 'pl_diff_log': np.log(pl_diff + 1)})
        #     ratings[i] = ratings[i].merge(local_df, on=["user"], how="left", sort=True)

        for i, epl in enumerate(self.exam_itempops):
            if i == len(self.exam_itempops) - 1:
                pop_diff = test_itempop[item_converters[i]] - epl
            else:
                pop_diff = itempop[item_converters[i]] - epl
            pop_diff = np.maximum(pop_diff, 0)
            local_df = pd.DataFrame(
                {'item': np.arange(len(pop_diff)), 'pop_diff': pop_diff, 'pop_diff_log': np.log(pop_diff + 1)})
            ratings[i] = ratings[i].merge(local_df, on=["item"], how="left", sort=False)

        for fold in range(self.n_folds + 2):
            test_pl_df = pd.DataFrame(
                {'user': np.arange(len(user_converters[fold])),
                 'pl': test_fpl[user_converters[fold]],
                 'pl_log': np.log(test_fpl + 1)[user_converters[fold]]})
            test_pop_df = pd.DataFrame({'item': np.arange(len(item_converters[fold])),
                                        'popularity': test_itempop[item_converters[fold]],
                                        'popularity_log': np.log(test_itempop + 1)[item_converters[fold]]})
            pl_df = pd.DataFrame({'user': np.arange(len(user_converters[fold])),
                                  'pl': fpl[user_converters[fold]],
                                  'pl_log': np.log(fpl + 1)[user_converters[fold]]})
            pop_df = pd.DataFrame(
                {'item': np.arange(len(item_converters[fold])),
                 'popularity': itempop[item_converters[fold]],
                 'popularity_log': np.log(itempop + 1)[item_converters[fold]]})
            if fold == self.n_folds + 1:
                ratings[fold] = ratings[fold].merge(test_pop_df, on=["item"], how="left", sort=False)
                ratings[fold] = ratings[fold].merge(test_pl_df, on=["user"], how="left", sort=True)
                #if self.exam_folder != folder:
                #    ratings[fold]["in_train"] = False
                #    compressed_urm = compress_urm(folder_test_urm, user_mapper, item_mapper,
                #                                  self.user_mappers[fold], self.item_mappers[fold])
                #    for index, row in ratings[fold].iterrows():
                #        if compressed_urm[row['user'], row['item']] != 0:
                #            ratings[fold].iloc[index, ratings[fold].columns.get_loc("in_train")] = True
            else:
                ratings[fold] = ratings[fold].merge(pop_df, on=["item"], how="left", sort=False)
                ratings[fold] = ratings[fold].merge(pl_df, on=["user"], how="left", sort=True)
                #if self.exam_folder != folder:
                #    ratings[fold]["in_train"] = False
                #    compressed_urm = compress_urm(folder_urm, user_mapper, item_mapper,
                #                                  self.user_mappers[fold], self.item_mappers[fold])
                #    for index, row in ratings[fold].iterrows():
                #        if compressed_urm[row['user'], row['item']] != 0:
                #            ratings[fold].iloc[index, ratings[fold].columns.get_loc("in_train")] = True

            if add_dataset_features_to_last_level_df:
                self.basic_dfs[fold] = self.basic_dfs[fold].merge(ratings[fold], on=["user", "item"], how="left",
                                                                  suffixes=('', '_' + folder), sort=True)

        for fold in range(self.n_folds + 2):
            fold_fpl = np.ediff1d(self.urms[fold].tocsr().indptr)
            fold_itempop = np.ediff1d(self.urms[fold].tocsc().indptr)
            pl_df = pd.DataFrame(
                {'user': np.arange(len(fold_fpl)), 'fold_pl': fold_fpl, 'fold_pl_log': np.log(fold_fpl + 1)})
            pop_df = pd.DataFrame({'item': np.arange(len(fold_itempop)), 'fold_popularity': fold_itempop,
                                   'fold_popularity_log': np.log(fold_itempop + 1)})
            ratings[fold] = ratings[fold].merge(pop_df, on=["item"], how="left", sort=False)
            ratings[fold] = ratings[fold].merge(pl_df, on=["user"], how="left", sort=True)

        return ratings


    def load_algorithms_predictions(self, folder):

        ratings = [pd.DataFrame({'user': self.validation_negs[fold].row.copy(),
                                 'item': self.validation_negs[fold].col.copy()})
                   for fold in range(len(self.validation_negs))]

        for algorithm in EXPERIMENTAL_CONFIG['baselines']:
            is_complete = True
            recname = algorithm.RECOMMENDER_NAME
            output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
            print("Loading", folder, recname)
            for fold in range(self.n_folds):
                validation_folder = self.exam_folder + "-" + str(fold)
                try:
                    r = row_minmax_scaling(
                        read_ratings(output_folder_path + validation_folder + "_valid_scores.tsv",
                                     self.user_mappers[fold], self.item_mappers[fold])).tocoo()
                    local_df = pd.DataFrame({'user': r.row, 'item': r.col, folder + "_" + recname: r.data})
                    ratings[fold] = ratings[fold].merge(local_df, on=["user", "item"], how="left", sort=True)
                except Exception as e:
                    is_complete = False
                    continue
            try:
                r = row_minmax_scaling(
                    read_ratings(output_folder_path + self.exam_folder + "_valid_scores.tsv",
                                 self.user_mappers[-2], self.item_mappers[-2])).tocoo()
                local_df = pd.DataFrame({'user': r.row, 'item': r.col, folder + "_" + recname: r.data})
                ratings[-2] = ratings[-2].merge(local_df, on=["user", "item"], how="left", sort=True)
            except Exception as e:
                is_complete = False
                continue
            try:
                r = row_minmax_scaling(
                    read_ratings(output_folder_path + self.exam_folder + "_test_scores.tsv",
                                 self.user_mappers[-1], self.item_mappers[-1])).tocoo()
                local_df = pd.DataFrame({'user': r.row, 'item': r.col, folder + "_" + recname: r.data})
                ratings[-1] = ratings[-1].merge(local_df, on=["user", "item"], how="left", sort=True)
            except Exception as e:
                is_complete = False
                continue

        return ratings


    def _load_ensemble_feature(self, folder, algo):

        results_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + \
                           "{}-ensemble-prediction-{}".format(algo, self.exam_folder)
        feat_name = '{}-{}-ensemble'.format(folder, algo)
        ratings = []

        if os.path.isfile(results_filename + "-valid.tsv.gz") and os.path.isfile(results_filename + "-test.tsv.gz"):

            for fold in range(self.n_folds):
                if os.path.isfile(results_filename + "-f{}.tsv.gz".format(fold)):
                    r = row_minmax_scaling(
                        read_ratings(results_filename + "-f{}.tsv.gz".format(fold),
                                     self.user_mappers[fold], self.item_mappers[fold])).tocoo()
                    ratings.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
                    self.basic_dfs[fold] = self.basic_dfs[fold].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

            r = row_minmax_scaling(
                read_ratings(results_filename + "-valid.tsv.gz", self.user_mappers[-2], self.item_mappers[-2])).tocoo()
            ratings.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
            self.basic_dfs[-2] = self.basic_dfs[-2].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

            r = row_minmax_scaling(
                read_ratings(results_filename + "-test.tsv.gz", self.user_mappers[-1], self.item_mappers[-1])).tocoo()
            ratings.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
            self.basic_dfs[-1] = self.basic_dfs[-1].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

        return ratings

    def load_lgbm_ensemble_feature(self, folder):
        return self._load_ensemble_feature(folder, "lgbm")

    def load_catboost_ensemble_feature(self, folder):
        return self._load_ensemble_feature(folder, "catboost")

    def load_xgb_ensemble_feature(self, folder):
        return self._load_ensemble_feature(folder, "xgb")

    def load_ratings_ensemble_feature(self, folder):
        # WARNING! ratings ensemble not available for local cv
        return self._load_ensemble_feature(folder, "ratings")

    def add_features(self, df, fold, on=["user", "item"], left_suffix="", right_suffix=""):
        self.basic_dfs[fold] = self.basic_dfs[fold].merge(
            df, on=on, how="left", suffixes=(left_suffix, right_suffix))

    def load_user_factors(self, folder, num_factors=16, epochs=25, reg=1e-4, normalize=False):
        # output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + algorithm.RECOMMENDER_NAME + os.sep
        # dataIO = DataIO(folder_path=output_folder_path)
        # data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")
        # hp = data_dict["hyperparameters_best"]
        folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper = self._prepare_folder_urms(folder)
        user_converters, item_converters = self._get_mapper_converters(user_mapper, item_mapper)
        hp = {"num_factors": num_factors, "epochs": epochs, "reg": reg}
        user_features, item_features = [], []
        for fold in range(len(self.urms)):
            urm = folder_urm
            if fold == len(self.urms) - 1:
                urm = folder_test_urm
            recommender = IALS(urm)
            recommender.fit(**hp)
            user_factors = recommender.USER_factors.astype(np.float32)[user_converters[fold]]
            item_factors = recommender.ITEM_factors.astype(np.float32)[item_converters[fold]]
            if normalize:
                user_factors = sklearn.preprocessing.normalize(user_factors, axis=1, norm="l2", copy=False)
                item_factors = sklearn.preprocessing.normalize(item_factors, axis=1, norm="l2", copy=False)
            uf_df = pd.DataFrame(user_factors, columns=["{}_uf_{}".format(folder, i) for i in range(num_factors)])
            uf_df["user"] = np.arange(len(uf_df))
            user_features.append(uf_df)
            uf_df = pd.DataFrame(item_factors, columns=["{}_if_{}".format(folder, i) for i in range(num_factors)])
            uf_df["item"] = np.arange(len(uf_df))
            item_features.append(uf_df)
            # self.basic_dfs[fold] = self.basic_dfs[fold].merge(uf_df, on=["user"], how="left", sort=True)
        return user_features, item_features

    def get_final_df(self):
        return self.basic_dfs.copy()

    def get_urms(self):
        return self.urms

    def get_user_mappers(self):
        return self.user_mappers

    def get_item_mappers(self):
        return self.item_mappers

    def get_validations(self):
        return self.validations

    def get_validation_negs(self):
        return self.validation_negs

