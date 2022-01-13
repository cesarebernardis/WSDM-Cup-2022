import os
import gzip
import shutil
import zipfile
import numpy as np
import scipy.sparse as sps
import pickle as pkl
import pandas as pd
import sklearn
import optuna

from scipy.stats import pearsonr

from RecSysFramework.Recommender.MatrixFactorization import IALS
from RecSysFramework.DataManager import Dataset
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


def add_noise(predictions, noise_weight=1e-3, random_seed=97):
    np.random.seed(random_seed)
    predictions.data += np.random.random(len(predictions.data)) * noise_weight
    return predictions


def reduce_score(predictions, perc=0.1):
    predictions = predictions.tocsr()
    pl = np.ediff1d(predictions.indptr)
    users_in_test = np.arange(len(pl))[pl > 0]
    users_to_delete = users_in_test[-int(perc * len(users_in_test)):]
    for u in users_to_delete:
        predictions.data[predictions.indptr[u]:predictions.indptr[u+1]] = 0.
    predictions.eliminate_zeros()
    return predictions


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
    urm = urm.tocsr()
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
        uid = user_mapping[split_row[0].strip()]
        iid = item_mapping[split_row[1].strip()]

        urm_row.append(uid)
        urm_col.append(iid)
        urm_data.append(float(split_row[2].strip()))

    file.close()

    return sps.csr_matrix((urm_data, (urm_row, urm_col)), shape=(n_users, n_items), dtype=np.float32)


def output_scores(filename, urm, user_mapper, item_mapper, user_prefix="", item_prefix="", compress=True):

    urm = urm.tocsr()
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


def evaluate(ratings, exam_test_urm, cutoff=10):
    scores = np.zeros(exam_test_urm.shape[0], dtype=np.float32)
    ratings = ratings.tocsr()
    exam_test_urm = exam_test_urm.tocsr()
    n_evals = 0
    for u in range(exam_test_urm.shape[0]):
        if exam_test_urm.indptr[u] != exam_test_urm.indptr[u + 1]:
            ranking = np.argsort(ratings.data[ratings.indptr[u]:ratings.indptr[u + 1]])[::-1]
            scores[u] = ndcg(ratings.indices[ratings.indptr[u]:ratings.indptr[u + 1]][ranking],
                             exam_test_urm.indices[exam_test_urm.indptr[u]:exam_test_urm.indptr[u + 1]], cutoff)
            n_evals += 1
    return scores, n_evals


def break_ties_with_filler(prediction, filler, penalization=1e-12, use_filler_ratings=False):
    prediction = prediction.tocsr()
    _filler = filler.tocsr(copy=True).astype(np.float32)
    rank_scores = np.arange(max(np.ediff1d(prediction.indptr))) * penalization
    assert prediction.shape == _filler.shape, "Prediction and filler have different shapes"
    for u in range(prediction.shape[0]):
        if _filler.indptr[u] != _filler.indptr[u+1]:
            start = _filler.indptr[u]
            end = _filler.indptr[u+1]
            if prediction.indptr[u] != prediction.indptr[u+1]:
                if use_filler_ratings:
                    _filler.data[start:end] *= penalization
                else:
                    ranking = np.argsort(_filler.data[start:end])
                    _filler.data[ranking + start] = rank_scores[:(end - start)]
            else:
                _filler.data[start:end] = 0.
    return prediction + _filler


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

    return urm_valid_total, urm_test_total


def remove_seen(predictions, seen):
    seen = seen.tocsr()
    predictions = predictions.tocsr()
    for u in range(seen.shape[0]):
        if predictions.indptr[u] != predictions.indptr[u+1] and seen.indptr[u] != seen.indptr[u+1]:
            pred_indices = predictions.indices[predictions.indptr[u]:predictions.indptr[u+1]]
            seen_indices = seen.indices[seen.indptr[u]:seen.indptr[u+1]]
            min_data = min(predictions.data[predictions.indptr[u]:predictions.indptr[u+1]])
            mask = np.isin(pred_indices, seen_indices, assume_unique=True)
            if sum(mask) > 0:
                predictions.data[np.arange(len(mask))[mask] + predictions.indptr[u]] = min_data - abs(min_data) * 0.02
    return predictions


def get_useless_columns(df, max_correlation=0.95, alpha=0.05):
    useless_cols = []
    columns = df.columns
    for i, column in enumerate(columns):
        vals = df[column].unique()
        if len(vals) < 2:
            useless_cols.append(column)
        else:
            for column2 in columns[i+1:]:
                r, p = pearsonr(df[column].to_numpy(), df[column2].to_numpy())
                if abs(r) >= max_correlation and p <= alpha:
                    useless_cols.append(column)
                    break
    return useless_cols


def remove_useless_features(df, columns_to_remove=None, inplace=True):
    if columns_to_remove is None:
        columns_to_remove = get_useless_columns(df)
    if len(columns_to_remove) > 0:
        df.drop(columns_to_remove, axis=1, inplace=inplace)



class FeatureGenerator:

    def __init__(self, exam_folder, n_folds=0, additional_negs=0):
        self.additional_negs = additional_negs
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
        exam_path = EXPERIMENTAL_CONFIG['dataset_folder'] + exam_folder + os.sep
        for fold in range(self.n_folds):
            validation_path = exam_path + exam_folder + "-" + str(
                fold) + os.sep
            with open(validation_path + "URM_all_train_mapper", "rb") as file:
                um, im = pkl.load(file)
                self.user_mappers.append(um)
                self.item_mappers.append(im)
            urm_train = sps.load_npz(validation_path + "URM_all_train.npz")
            self.urms.append(urm_train)
            self.exam_itempops.append(np.ediff1d(urm_train.tocsc().indptr))
            self.validations.append(sps.load_npz(validation_path + "URM_all_test.npz").tocoo())
            self.validation_negs.append([load_compressed_csr_matrix(validation_path + "urm_neg.npz").tocoo()] +
                [load_compressed_csr_matrix(validation_path + "urm_neg_{}.npz".format(i)).tocoo() for i in range(self.additional_negs)])

        self.user_mappers.append(exam_user_mapper)
        self.item_mappers.append(exam_item_mapper)
        self.validation_negs.append([self.urm_exam_valid_neg.tocoo()] +
            [load_compressed_csr_matrix(exam_path + "urm_valid_neg_{}.npz".format(i)).tocoo() for i in range(self.additional_negs)])
        self.validations.append(self.exam_valid.get_URM().tocoo())
        self.urms.append(self.exam_train.get_URM())
        self.exam_profile_lengths.append(np.ediff1d(self.urms[-1].indptr))
        self.exam_itempops.append(np.ediff1d(self.urms[-1].tocsc().indptr))

        self.user_mappers.append(exam_user_mapper)
        self.item_mappers.append(exam_item_mapper)
        self.validation_negs.append([self.urm_exam_test_neg.tocoo()])
        valid_urm = self.exam_valid.get_URM()
        valid_urm.data[:] = max(self.urms[-1].data)
        self.urms.append(self.exam_train.get_URM() + valid_urm)
        self.exam_profile_lengths.append(np.ediff1d(self.urms[-1].indptr))
        self.exam_itempops.append(np.ediff1d(self.urms[-1].tocsc().indptr))

        self.basic_dfs = []
        for fold in range(self.n_folds + 2):
            self.basic_dfs.append(self._initialize_dataframe(fold))
            fold_fpl = np.ediff1d(self.urms[fold].tocsr().indptr)
            fold_itempop = np.ediff1d(self.urms[fold].tocsc().indptr)
            pl_df = pd.DataFrame({'user': np.arange(len(fold_fpl)), 'pl': fold_fpl, 'pl_log': np.log(fold_fpl + 1)})
            pop_df = pd.DataFrame({'item': np.arange(len(fold_itempop)), 'popularity': fold_itempop,
                                   'popularity_log': np.log(fold_itempop + 1)})
            self.basic_dfs[fold] = self.basic_dfs[fold].merge(pop_df, on=["item"], how="left", sort=True)
            self.basic_dfs[fold] = self.basic_dfs[fold].merge(pl_df, on=["user"], how="left", sort=True)


    def _initialize_dataframe(self, fold):
        rows = np.concatenate([self.validation_negs[fold][i].row for i in range(len(self.validation_negs[fold]))])
        cols = np.concatenate([self.validation_negs[fold][i].col for i in range(len(self.validation_negs[fold]))])
        return pd.DataFrame({'user': rows, 'item': cols})


    def _get_mapper_converters(self, user_mapper, item_mapper):

        user_converters = []
        for i in range(len(self.user_mappers)):
            converter = np.zeros(len(self.user_mappers[i]), dtype=np.int32)
            inv_mapper = {v: k for k, v in self.user_mappers[i].items()}
            for k, v in inv_mapper.items():
                converter[k] = user_mapper[v]
            user_converters.append(converter)
            assert len(np.unique(converter)) == len(inv_mapper), "Error user: {} - {}".format(len(np.unique(converter)), len(inv_mapper))

        item_converters = []
        for i in range(len(self.item_mappers)):
            converter = np.zeros(len(self.item_mappers[i]), dtype=np.int32)
            inv_mapper = {v: k for k, v in self.item_mappers[i].items()}
            for k, v in inv_mapper.items():
                converter[k] = item_mapper[v]
            item_converters.append(converter)
            assert len(np.unique(converter)) == len(inv_mapper), "Error item: {} - {}".format(len(np.unique(converter)), len(inv_mapper))

        return user_converters, item_converters


    def _prepare_folder_urms(self, folder):
        folder_train, folder_valid, _, _ = create_dataset_from_folder(folder)
        user_mapper, item_mapper = folder_train.get_URM_mapper()
        folder_urm = folder_train.get_URM()
        folder_valid_urm = folder_valid.get_URM()
        folder_valid_urm.data[:] = max(folder_urm.data)
        folder_test_urm = folder_urm + folder_valid_urm
        return folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper


    def load_folder_features(self, folder, include_fold_features=True):

        folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper = self._prepare_folder_urms(folder)

        fpl = np.ediff1d(folder_urm.tocsr().indptr)
        itempop = np.ediff1d(folder_urm.tocsc().indptr)
        item_avg_rating = np.divide(np.array(folder_urm.sum(axis=0)).flatten(), 1e-6 + itempop)
        user_avg_rating = np.divide(np.array(folder_urm.sum(axis=1)).flatten(), 1e-6 + fpl)

        test_fpl = np.ediff1d(folder_test_urm.tocsr().indptr)
        test_itempop = np.ediff1d(folder_test_urm.tocsc().indptr)
        test_item_avg_rating = np.divide(np.array(folder_test_urm.sum(axis=0)).flatten(), 1e-6 + test_itempop)
        test_user_avg_rating = np.divide(np.array(folder_test_urm.sum(axis=1)).flatten(), 1e-6 + test_fpl)

        ratings = [self._initialize_dataframe(fold) for fold in range(len(self.validation_negs))]

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
                {'item': np.arange(len(pop_diff)), folder + '_pop_diff': pop_diff, folder + '_pop_diff_log': np.log(pop_diff + 1)})
            ratings[i] = ratings[i].merge(local_df, on=["item"], how="left", suffixes=("", "_" + folder), sort=False)

        for fold in range(self.n_folds + 2):
            if fold == self.n_folds + 1:
                pl_df = pd.DataFrame({'user': np.arange(len(user_converters[fold])),
                                      folder + '_uar': test_user_avg_rating[user_converters[fold]],
                                      folder + '_pl': test_fpl[user_converters[fold]],
                                      folder + '_pl_log': np.log(test_fpl + 1)[user_converters[fold]]})
                pop_df = pd.DataFrame({'item': np.arange(len(item_converters[fold])),
                                       folder + '_iar': test_item_avg_rating[item_converters[fold]],
                                       folder + '_popularity': test_itempop[item_converters[fold]],
                                       folder + '_popularity_log': np.log(test_itempop + 1)[item_converters[fold]]})
            else:
                pl_df = pd.DataFrame({'user': np.arange(len(user_converters[fold])),
                                      folder + '_uar': user_avg_rating[user_converters[fold]],
                                      folder + '_pl': fpl[user_converters[fold]],
                                      folder + '_pl_log': np.log(fpl + 1)[user_converters[fold]]})
                pop_df = pd.DataFrame({'item': np.arange(len(item_converters[fold])),
                                       folder + '_iar': item_avg_rating[item_converters[fold]],
                                       folder + '_popularity': itempop[item_converters[fold]],
                                       folder + '_popularity_log': np.log(itempop + 1)[item_converters[fold]]})

            ratings[fold] = ratings[fold].merge(pop_df, on=["item"], how="left", sort=False)
            ratings[fold] = ratings[fold].merge(pl_df, on=["user"], how="left", sort=True)

        if include_fold_features:
            for fold in range(self.n_folds + 2):
                fold_fpl = np.ediff1d(self.urms[fold].tocsr().indptr)
                fold_itempop = np.ediff1d(self.urms[fold].tocsc().indptr)
                pl_df = pd.DataFrame(
                    {'user': np.arange(len(fold_fpl)), folder + '_fold_pl': fold_fpl, folder + '_fold_pl_log': np.log(fold_fpl + 1)})
                pop_df = pd.DataFrame({'item': np.arange(len(fold_itempop)), folder + '_fold_popularity': fold_itempop,
                                       folder + '_fold_popularity_log': np.log(fold_itempop + 1)})
                ratings[fold] = ratings[fold].merge(pop_df, on=["item"], how="left", sort=False)
                ratings[fold] = ratings[fold].merge(pl_df, on=["user"], how="left", sort=True)

        return ratings


    def load_algorithms_predictions(self, folder, normalize=True, only_best_baselines=False):

        ratings = [self._initialize_dataframe(fold) for fold in range(len(self.validation_negs))]

        if only_best_baselines:
            algorithms = EXPERIMENTAL_CONFIG['best-baselines']
        else:
            algorithms = EXPERIMENTAL_CONFIG['baselines']

        for algorithm in algorithms:
            is_complete = True
            recname = algorithm.RECOMMENDER_NAME
            output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
            if not os.path.isfile(output_folder_path + self.exam_folder + "_valid_scores.tsv.gz") or \
                        not os.path.isfile(output_folder_path + self.exam_folder + "_test_scores.tsv.gz"):
                print(recname, folder, "predictions not found")
                continue
            print("Loading", folder, recname)
            featname = folder + "_" + recname
            if normalize:
                featname += "_norm"
            for fold in range(self.n_folds):
                validation_folder = self.exam_folder + "-" + str(fold)
                dfs = []
                for i in range(-1, self.additional_negs):
                    suffix = "" if i < 0 else "-{}".format(i)
                    r = read_ratings(output_folder_path + validation_folder + "_valid_scores{}.tsv.gz".format(suffix),
                                     self.user_mappers[fold], self.item_mappers[fold])
                    if normalize:
                        r = row_minmax_scaling(r)
                    r = r.tocoo()
                    dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, featname: r.data}))
                ratings[fold] = ratings[fold].merge(pd.concat(dfs), on=["user", "item"], how="left", sort=True)

            dfs = []
            for i in range(-1, self.additional_negs):
                suffix = "" if i < 0 else "-{}".format(i)
                r = read_ratings(output_folder_path + self.exam_folder + "_valid_scores{}.tsv.gz".format(suffix),
                                 self.user_mappers[-2], self.item_mappers[-2])
                if normalize:
                    r = row_minmax_scaling(r)
                r = r.tocoo()
                dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, featname: r.data}))
            ratings[-2] = ratings[-2].merge(pd.concat(dfs), on=["user", "item"], how="left", sort=True)

            dfs = []
            for i in range(-1, self.additional_negs):
                suffix = "" if i < 0 else "-{}".format(i)
                r = read_ratings(output_folder_path + self.exam_folder + "_test_scores{}.tsv.gz".format(suffix),
                                 self.user_mappers[-1], self.item_mappers[-1])
                if normalize:
                    r = row_minmax_scaling(r)
                r = r.tocoo()
                dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, featname: r.data}))
            ratings[-1] = ratings[-1].merge(pd.concat(dfs), on=["user", "item"], how="left", sort=True)

        return ratings


    def _load_ensemble_feature(self, folder, algo, normalize=True, break_ties=False, break_ties_penalization=1e-4):

        results_basepath = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep

        results_filename = results_basepath + "{}-ensemble-prediction-{}".format(algo, self.exam_folder)
        break_ties_filename = results_basepath + "ratings-ensemble-prediction-{}".format(self.exam_folder)
        feat_name = '{}-{}-ensemble'.format(folder, algo)
        if normalize:
            feat_name += "-norm"
        ratings = []

        if os.path.isfile(results_filename + "-valid.tsv.gz") and os.path.isfile(results_filename + "-test.tsv.gz"):

            for fold in range(self.n_folds):
                dfs = []
                for i in range(-1, self.additional_negs):
                    suffix = "" if i < 0 else "-{}".format(i)
                    r = read_ratings(results_filename + "-f{}{}.tsv.gz".format(fold, suffix),
                                     self.user_mappers[fold], self.item_mappers[fold])
                    if normalize:
                        r = row_minmax_scaling(r)
                    r = r.tocoo()
                    if break_ties:
                        filler = row_minmax_scaling(read_ratings(break_ties_filename + "-f{}{}.tsv.gz".format(fold, suffix), self.user_mappers[fold], self.item_mappers[fold]))
                        r = break_ties_with_filler(r, filler, use_filler_ratings=True, penalization=break_ties_penalization).tocoo()
                    dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
                ratings.append(pd.concat(dfs))
                self.basic_dfs[fold] = self.basic_dfs[fold].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

            dfs = []
            for i in range(-1, self.additional_negs):
                suffix = "" if i < 0 else "-{}".format(i)
                r = read_ratings(results_filename + "-valid{}.tsv.gz".format(suffix), self.user_mappers[-2], self.item_mappers[-2])
                if normalize:
                    r = row_minmax_scaling(r)
                r = r.tocoo()
                if break_ties:
                    filler = row_minmax_scaling(read_ratings(break_ties_filename + "-valid{}.tsv.gz".format(suffix), self.user_mappers[-2], self.item_mappers[-2]))
                    r = break_ties_with_filler(r, filler, use_filler_ratings=True, penalization=break_ties_penalization).tocoo()
                dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
            ratings.append(pd.concat(dfs))
            self.basic_dfs[-2] = self.basic_dfs[-2].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

            dfs = []
            for i in range(-1, self.additional_negs):
                suffix = "" if i < 0 else "-{}".format(i)
                r = read_ratings(results_filename + "-test{}.tsv.gz".format(suffix), self.user_mappers[-1], self.item_mappers[-1])
                if normalize:
                    r = row_minmax_scaling(r)
                r = r.tocoo()
                if break_ties:
                    filler = row_minmax_scaling(read_ratings(break_ties_filename + "-test{}.tsv.gz".format(suffix), self.user_mappers[-1], self.item_mappers[-1]))
                    r = break_ties_with_filler(r, filler, use_filler_ratings=True, penalization=break_ties_penalization).tocoo()
                dfs.append(pd.DataFrame({'user': r.row, 'item': r.col, feat_name: r.data}))
            ratings.append(pd.concat(dfs))
            self.basic_dfs[-1] = self.basic_dfs[-1].merge(ratings[-1], on=["user", "item"], how="left", sort=True)

        return ratings

    def load_lgbm_ensemble_feature(self, folder, normalize=True, break_ties=False, break_ties_penalization=1e-4):
        return self._load_ensemble_feature(folder, "lgbm", normalize=normalize, break_ties=break_ties, break_ties_penalization=break_ties_penalization)

    def load_catboost_ensemble_feature(self, folder, normalize=True, break_ties=False, break_ties_penalization=1e-4):
        return self._load_ensemble_feature(folder, "catboost", normalize=normalize, break_ties=break_ties, break_ties_penalization=break_ties_penalization)

    def load_xgb_ensemble_feature(self, folder, normalize=True, break_ties=False, break_ties_penalization=1e-4):
        return self._load_ensemble_feature(folder, "xgb", normalize=normalize, break_ties=break_ties, break_ties_penalization=break_ties_penalization)

    def load_pyltr_ensemble_feature(self, folder, normalize=True, break_ties=False, break_ties_penalization=1e-4):
        return self._load_ensemble_feature(folder, "pyltr", normalize=normalize, break_ties=break_ties, break_ties_penalization=break_ties_penalization)

    def load_ratings_ensemble_feature(self, folder, normalize=True, break_ties=False, break_ties_penalization=1e-4):
        # WARNING! ratings ensemble not available for local cv
        return self._load_ensemble_feature(folder, "ratings", normalize=normalize, break_ties=break_ties, break_ties_penalization=break_ties_penalization)

    def add_features(self, df, fold, on=["user", "item"], left_suffix="", right_suffix=""):
        self.basic_dfs[fold] = self.basic_dfs[fold].merge(
            df, on=on, how="left", suffixes=(left_suffix, right_suffix))

    def load_user_factors(self, folder, num_factors=16, epochs=25, reg=1e-4, normalize=True, use_val_factors_for_test=False):
        # output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + algorithm.RECOMMENDER_NAME + os.sep
        # dataIO = DataIO(folder_path=output_folder_path)
        # data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")
        # hp = data_dict["hyperparameters_best"]
        folder_urm, folder_valid_urm, folder_test_urm, user_mapper, item_mapper = self._prepare_folder_urms(folder)
        user_converters, item_converters = self._get_mapper_converters(user_mapper, item_mapper)
        hp = {"num_factors": num_factors, "epochs": epochs, "reg": reg}
        user_features, item_features = [], []
        for fold in range(len(self.urms)):
            if fold == len(self.urms) - 1 and use_val_factors_for_test:
                user_features.append(user_features[-1].copy())
                item_features.append(item_features[-1].copy())
            else:
                urm = folder_urm
                if fold == len(self.urms) - 1:
                    urm = folder_test_urm
                recommender = IALS(urm)
                recommender.fit(**hp)
                user_factors = recommender.USER_factors.astype(np.float32)[user_converters[fold]]
                item_factors = recommender.ITEM_factors.astype(np.float32)[item_converters[fold]]
                if normalize:
                    user_factors_norms = np.linalg.norm(user_factors, axis=1)
                    item_factors_norms = np.linalg.norm(item_factors, axis=1)
                    user_factors = sklearn.preprocessing.normalize(user_factors, axis=1, norm="l2", copy=False)
                    item_factors = sklearn.preprocessing.normalize(item_factors, axis=1, norm="l2", copy=False)
                uf_df = pd.DataFrame(user_factors, columns=["{}_uf_{}".format(folder, i) for i in range(num_factors)])
                uf_df["user"] = np.arange(len(uf_df))
                if normalize:
                    uf_df["{}_uf_norm".format(folder)] = user_factors_norms
                user_features.append(uf_df)
                uf_df = pd.DataFrame(item_factors, columns=["{}_if_{}".format(folder, i) for i in range(num_factors)])
                uf_df["item"] = np.arange(len(uf_df))
                if normalize:
                    uf_df["{}_if_norm".format(folder)] = item_factors_norms
                item_features.append(uf_df)

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



class Optimizer:

    VAL_WEIGHT = 2.
    NAME = "Generic"

    def __init__(self, urms, ratings, validations, fillers=None, n_folds=5, random_trials_perc=0.25):
        self.urms = urms
        self.ratings = ratings
        self.validations = validations
        self.n_folds = n_folds
        if fillers is None:
            self.fillers = [None for _ in range(len(validations))]
        else:
            self.fillers = []
            for filler in fillers:
                f = row_minmax_scaling(filler)
                scores = np.arange(max(np.ediff1d(f.indptr)))
                scores = np.around(scores / (scores[-1] + 1), decimals=1)
                for u in range(f.shape[0]):
                    start = f.indptr[u]
                    end = f.indptr[u+1]
                    if start != end:
                        ranking = np.argsort(f.data[start:end])
                        f.data[start + ranking] = scores[-(end - start):]
                self.fillers.append(f)
        self.best_params = None
        self.random_trials_perc = random_trials_perc

    def get_splits(self, ratings, validation, random_state=1007):
        splitter = RandomizedGroupKFold(self.n_folds, random_state=random_state)
        return [(tr_i, te_i) for tr_i, te_i in splitter.split(ratings, validation, ratings.user.values)]

    def get_params_from_trial(self, trial):
        pass

    def train_cv(self, _params, _urm, _ratings, _validation, test_df=None, filler=None):
        pass

    def train_cv_best_params(self, _urm, _ratings, _validation, test_df=None, filler=None):
        return self.train_cv(self.best_params, _urm, _ratings, _validation, test_df=test_df, filler=filler)

    def objective(self, trial):
        final_score = 0.
        denominator = 0.
        params = self.get_params_from_trial(trial)
        for fold in range(len(self.validations)):
            if fold == len(self.validations) - 1:
                _val_weight = self.VAL_WEIGHT
            else:
                _val_weight = 1.
            _, part_score = self.train_cv(params, self.urms[fold], self.ratings[fold], self.validations[fold],
                                          filler=self.fillers[fold])
            final_score += part_score * _val_weight
            denominator += _val_weight
        return final_score / denominator


    def optimize(self, study_name, storage, force=False, n_trials=250):
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(self.random_trials_perc * n_trials))
        if force:
            try:
                optuna.study.delete_study(study_name, storage)
            except KeyError:
                pass
        study = optuna.create_study(direction="maximize", study_name=study_name,
                    sampler=sampler, load_if_exists=True, storage=storage)
        if n_trials - len(study.trials) > 0:
            study.optimize(self.objective, n_trials=n_trials - len(study.trials), show_progress_bar=True)
        return study


    def optimize_all(self, exam_folder, force=False, n_trials=250, folder=None, study_name_suffix=""):
        result_dict = {}
        storage = "sqlite:///" + EXPERIMENTAL_CONFIG['dataset_folder']
        if folder is not None:
            storage += folder + os.sep + "optuna.db"
        else:
            storage += exam_folder + os.sep + "optuna.db"
        if folder is not None:
            study_name = "{}-ensemble-{}-{}{}".format(self.NAME, folder, exam_folder, study_name_suffix)
        else:
            study_name = "{}-last-level-ensemble-{}{}".format(self.NAME, exam_folder, study_name_suffix)
        study = self.optimize(study_name, storage, force=force, n_trials=n_trials)
        self.best_params = study.best_params
        return self.best_params



from sklearn.model_selection import GroupKFold
from sklearn.utils import check_array

class RandomizedGroupKFold(GroupKFold):

    """
    The original GroupKFold does not assume balance in the sizes of the groups, so it does not shuffle
    and it tries to balance the splits as much as possible.
    In this case, groups are balanced, so no need to do it, instead I would like to shuffle.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        super().__init__(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        samples_per_fold = int(len(unique_groups) / self.n_splits)
        indices = np.repeat(np.arange(self.n_splits), samples_per_fold)
        indices = np.concatenate([indices, np.arange(len(unique_groups) - len(indices))])

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.shuffle:
            np.random.shuffle(indices)

        for f in range(self.n_splits):
            yield np.asarray(np.isin(unique_groups[groups], unique_groups[indices == f])).nonzero()[0]

