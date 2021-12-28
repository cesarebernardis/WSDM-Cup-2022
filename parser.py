import os.path
import pickle as pkl
import numpy as np
import scipy.sparse as sps
import gzip


def open_file(filename):
    file = None
    if os.path.isfile(filename):
        file = open(filename, "r")
    elif os.path.isfile(filename + ".gz"):
        file = gzip.open(filename + ".gz", "rt")
    return file


def read_parse_save(folder, threshold=0., binarize=False, force_user_merge=False):

    print("Working on {}".format(folder))

    has_test = os.path.isfile(folder + "/test_run.tsv") or os.path.isfile(folder + "/test_run.tsv.gz")

    user_mapping = {}
    item_mapping = {}

    urm_row = []
    urm_col = []
    urm_data = []
    urm_binary_data = []

    file = open_file(folder + "/train.tsv")

    if file is not None:

        jump_header = True
        for line in file:

            if jump_header:
                jump_header = False
                continue

            split_row = line.strip().split("\t")
            uname = split_row[0].strip()
            iname = split_row[1].strip()
            rating = float(split_row[2].strip())
            if force_user_merge:
                uname = uname[2:]

            if rating < threshold:
                rating = 0.

            if uname in user_mapping.keys():
                uid = user_mapping[uname]
            else:
                uid = len(user_mapping)
                user_mapping[uname] = uid

            if iname in item_mapping.keys():
                iid = item_mapping[iname]
            else:
                iid = len(item_mapping)
                item_mapping[iname] = iid

            if rating > 0.:
                urm_row.append(uid)
                urm_col.append(iid)
                urm_data.append(rating)
                urm_binary_data.append(1.)

        file.close()

    file = open_file(folder + "/train_5core.tsv")

    if file is not None:

        jump_header = True
        for line in file:

            if jump_header:
                jump_header = False
                continue

            split_row = line.strip().split("\t")
            uname = split_row[0].strip()
            iname = split_row[1].strip()
            if force_user_merge:
                uname = uname[2:]

            if uname in user_mapping.keys():
                uid = user_mapping[uname]
            else:
                uid = len(user_mapping)
                user_mapping[uname] = uid

            if iname in item_mapping.keys():
                iid = item_mapping[iname]
            else:
                iid = len(item_mapping)
                item_mapping[iname] = iid

            urm_row.append(uid)
            urm_col.append(iid)
            urm_data.append(4.)
            urm_binary_data.append(1.)

        file.close()

    urm_valid_row = []
    urm_valid_col = []
    urm_valid_data = []
    userset = set()


    file = open_file(folder + "/valid_qrel.tsv")

    if file is not None:

        jump_header = True
        for line in file:

            if jump_header:
                jump_header = False
                continue

            split_row = line.strip().split("\t")
            uname = split_row[0].strip()
            iname = split_row[1].strip()
            if force_user_merge:
                uname = uname[2:]

            if uname in user_mapping.keys():
                uid = user_mapping[uname]
            else:
                uid = len(user_mapping)
                user_mapping[uname] = uid

            if iname in item_mapping.keys():
                iid = item_mapping[iname]
            else:
                iid = len(item_mapping)
                item_mapping[iname] = iid

            if uid in userset:
                # Only one interaction in validation/test per user
                continue
            userset.add(uid)

            urm_valid_row.append(uid)
            urm_valid_col.append(iid)
            urm_valid_data.append(float(split_row[2].strip()))

        file.close()

    urm_valid_neg_row = []
    urm_valid_neg_col = []
    userset = set()

    file = open_file(folder + "/valid_run.tsv")

    if file is not None:

        for line in file:

            split_row = line.strip().split("\t")
            uname = split_row[0].strip()
            if force_user_merge:
                uname = uname[2:]

            if uname in user_mapping.keys():
                uid = user_mapping[uname]
            else:
                uid = len(user_mapping)
                user_mapping[uname] = uid

            if uid in userset:
                # Only one interaction in validation/test per user
                continue
            userset.add(uid)

            for iname in split_row[1].strip().split(","):

                iname = iname.strip()
                if iname in item_mapping.keys():
                    iid = item_mapping[iname]
                else:
                    iid = len(item_mapping)
                    item_mapping[iname] = iid

                urm_valid_neg_row.append(uid)
                urm_valid_neg_col.append(iid)

        file.close()

    if has_test:

        urm_test_neg_row = []
        urm_test_neg_col = []

        file = open_file(folder + "/test_run.tsv")

        if file is not None:

            for line in file:

                split_row = line.strip().split("\t")
                uname = split_row[0].strip()
                if force_user_merge:
                    uname = uname[2:]

                if uname in user_mapping.keys():
                    uid = user_mapping[uname]
                else:
                    uid = len(user_mapping)
                    user_mapping[uname] = uid

                for iname in split_row[1].split(","):

                    iname = iname.strip()
                    if iname in item_mapping.keys():
                        iid = item_mapping[iname]
                    else:
                        iid = len(item_mapping)
                        item_mapping[iname] = iid

                    urm_test_neg_row.append(uid)
                    urm_test_neg_col.append(iid)

            file.close()

    n_users = len(user_mapping)
    n_items = len(item_mapping)

    urm = sps.csr_matrix((urm_data, (urm_row, urm_col)), shape=(n_users, n_items), dtype=np.float32)
    urm.eliminate_zeros()
    urm.sort_indices()

    binary_urm = sps.csr_matrix((urm_binary_data, (urm_row, urm_col)), shape=(n_users, n_items), dtype=np.float32)
    binary_urm.eliminate_zeros()
    binary_urm.sort_indices()
    binary_urm = binary_urm.tocoo()

    for i in range(len(binary_urm.data)):
        if binary_urm.data[i] > 1.:
            urm[binary_urm.row[i], binary_urm.col[i]] /= binary_urm.data[i]

    del binary_urm

    if binarize:
        urm.data[:] = 1.
    print("urm", urm.shape, len(urm.data))
    sps.save_npz(folder + "/urm.npz", urm, compressed=True)

    urm_valid = sps.csr_matrix((urm_valid_data, (urm_valid_row, urm_valid_col)), shape=(n_users, n_items), dtype=np.float32)
    urm_valid.eliminate_zeros()
    urm_valid.sort_indices()
    if binarize:
        urm_valid.data[:] = 1.
    print("urm_valid", urm_valid.shape, len(urm_valid.data))
    sps.save_npz(folder + "/urm_valid.npz", urm_valid, compressed=True)

    urm_valid_neg = sps.csr_matrix((np.ones(len(urm_valid_neg_row)), (urm_valid_neg_row, urm_valid_neg_col)), shape=(n_users, n_items), dtype=np.float32)
    urm_valid_neg.eliminate_zeros()
    urm_valid_neg.sort_indices()
    print("urm_valid_neg", urm_valid_neg.shape, len(urm_valid_neg.data))
    sps.save_npz(folder + "/urm_valid_neg.npz", urm_valid_neg, compressed=True)

    # Small check: all positive items in validation have to be in items to rank for that user
    users_to_recommend = np.arange(urm_valid.shape[0])[np.ediff1d(urm_valid.indptr) > 0]
    for u in users_to_recommend:
        assert np.all(np.isin(
            urm_valid.indices[urm_valid.indptr[u]:urm_valid.indptr[u+1]],
            urm_valid_neg.indices[urm_valid_neg.indptr[u]:urm_valid_neg.indptr[u+1]]
        )), "Validation item for user {} not in items to rank".format(u)

    if has_test:
        urm_test_neg = sps.csr_matrix((np.ones(len(urm_test_neg_row)), (urm_test_neg_row, urm_test_neg_col)), shape=(n_users, n_items), dtype=np.float32)
        urm_test_neg.eliminate_zeros()
        urm_test_neg.sort_indices()
        print("urm_test_neg", urm_test_neg.shape, len(urm_test_neg.data))
        sps.save_npz(folder + "/urm_test_neg.npz", urm_test_neg, compressed=True)

    with open(folder + "/user_mapping.pkl", "wb") as handle:
        pkl.dump(user_mapping, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(folder + "/item_mapping.pkl", "wb") as handle:
        pkl.dump(item_mapping, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print("-----------------")


if __name__ == "__main__":
    for subdir in os.listdir("datasets"):
        dir = "datasets" + os.sep + subdir
        if os.path.isdir(dir) and ("t1" in dir or "t2" in dir):
            read_parse_save(dir)
