#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
import scipy.sparse as sps

from cython.parallel import prange

cimport numpy as np

from libc.math cimport exp, sqrt
from libc.stdlib cimport srand, rand, RAND_MAX



cdef int randint(int max):
    return int(float(rand()) / RAND_MAX * max)


cdef void permute(int[:] r, int rsize):

    cdef int i, j, temp

    for i in range(rsize):
        #j = randint(rsize-i-1) + i
        j = (rand() % (rsize-i)) + i
        temp = r[i]
        r[i] = r[j]
        r[j] = temp


def generate_test_negative_for_user(int[:] train_interactions, int[:] test_interactions, int samples, int n_items, int random_seed=-1):

    cdef int i, j
    cdef int[:] all_items = np.arange(n_items, dtype=np.int32)
    cdef int[:] active = np.ones(n_items, dtype=np.int32)
    cdef int n_interactions = len(train_interactions) + len(test_interactions)

    if random_seed >= 0:
        srand(random_seed)

    if n_items - n_interactions < samples:
        print("User has not the required number of negative samples ({}/{})"
              .format(n_items - n_interactions, samples))
        samples = n_items - n_interactions

    cdef int[:] negatives = np.empty(samples, dtype=np.int32)

    permute(all_items, n_items)
    for i in range(len(train_interactions)):
        active[train_interactions[i]] = 0
    for i in range(len(test_interactions)):
        active[test_interactions[i]] = 0

    i = 0
    j = 0
    while j < samples:
        if active[all_items[i]]:
            negatives[j] = all_items[i]
            j += 1
        i += 1

    return np.array(negatives)



def generate_URM_test_negative(URM_train, URM_test, negative_samples=100, type="fixed", recommenders=None,
                               recommenders_weights=None, holdout_perc=0., popularity_proportional="", random_seed=42):

    type = type.lower()
    assert type in ["fixed", "per-positive", "balanced"], "Unknown negative sampling type \"{}\"".format(type)

    URM = (URM_train + URM_test).tocsr()

    cdef int n_interactions, samples, i, j, user
    cdef int ubatch_idx, batch_size
    cdef int[:] interactions

    cdef float _holdout_perc = holdout_perc
    cdef int n_items = URM.shape[1]
    cdef int n_users = URM.shape[0]
    cdef int maxu, missing_samples
    cdef int[:] old_cutoff, cutoff
    cdef float[:] density_backup, density
    cdef float[:, :] similarity
    cdef int[:] indptr = np.zeros(n_users + 1, dtype=np.int32)
    cdef int[:] urm_indptr = URM.indptr
    cdef int[:] urm_indices = URM.indices
    cdef int[:] urm_test_indptr = URM_test.tocsr().indptr
    cdef int[:] urm_test_indices = URM_test.tocsr().indices
    cdef float[:] rec_candidates_perc
    cdef int isrecbased = 0

    indices = [None] * n_users
    batch_size = 5000

    if popularity_proportional == "_popprop":
        density = np.ediff1d(URM_train.tocsc().indptr).astype(np.float32)
    else:
        density = np.ones(n_items, dtype=np.float32)

    density_backup = density.copy()

    srand(random_seed)
    if "_recbased" in popularity_proportional:
        isrecbased = 1
        if recommenders_weights is None:
            recommenders_weights = np.ones(len(recommenders), dtype=np.float32)
        recommenders_weights = np.array(recommenders_weights, dtype=np.float32)
        cutoff = np.zeros(len(recommenders), dtype=np.int32)
        if recommenders_weights.ndim == 1:
            rec_candidates_perc = recommenders_weights / sum(recommenders_weights)
        else:
            recommenders_weights = recommenders_weights.T
            if recommenders_weights.shape[0] != n_users:
                raise Exception("Incorrect number of values provided in the user-wise recommender weights ({}/{})"
                                .format(recommenders_weights.shape[1], n_users))

    for user in range(n_users):

        n_interactions = urm_indptr[user+1] - urm_indptr[user]
        n_test_interactions = urm_test_indptr[user+1] - urm_test_indptr[user]
        interactions = urm_indices[urm_indptr[user]:urm_indptr[user+1]]
        samples = 0

        if user % batch_size == 0:
            ubatch_idx = 0
            maxu = min(user+batch_size, n_users)
            if popularity_proportional == "_compprop":
                similarity = (URM_train.T[urm_test_indices[user:maxu], :].dot(URM_train).toarray() + 1).astype(np.float32)
            elif isrecbased:
                recommendations = [recommender.recommend(np.arange(n_users)[user:maxu],
                    remove_seen_flag=True, cutoff=negative_samples+1, remove_top_pop_flag=False,
                    remove_custom_items_flag=False, return_scores=False) for recommender in recommenders]

        if n_interactions > 0 and n_test_interactions > 0:

            if type == "fixed":
                samples = negative_samples
            elif type == "balanced":
                if _holdout_perc > 0.:
                    samples = max(1, round(_holdout_perc * (n_items - n_interactions)))
                else:
                    samples = max(1, round((n_items - n_interactions) / n_interactions))
            else:
                samples = n_test_interactions * negative_samples

            if n_items - n_interactions < samples:
                print("User has not the required number of negative samples ({}/{})"
                      .format(n_items - n_interactions, samples))
                samples = n_items - n_interactions

            if popularity_proportional == "_compprop":
                density = similarity[ubatch_idx, :]

            if isrecbased:

                tmp = np.array([])
                old_cutoff = np.zeros(len(recommenders), dtype=np.int32)
                if recommenders_weights.ndim == 2:
                    rw = recommenders_weights[user, :] + 1e-8
                    rec_candidates_perc = rw / sum(rw)
                while len(tmp) < samples:
                    missing_samples = samples - len(tmp)
                    for i in range(len(recommenders)):
                        cutoff[i] = old_cutoff[i] + max(1, np.ceil(missing_samples * rec_candidates_perc[i]))
                    tmp = np.setdiff1d(
                        np.concatenate(
                            [tmp] + [np.array(recs[ubatch_idx])[old_cutoff[i]:cutoff[i]]
                                     for i, recs in enumerate(recommendations)]
                        ), urm_test_indices[user]
                    )
                    for i in range(len(recommenders)):
                        old_cutoff[i] = cutoff[i]

                indices[user] = tmp[:samples].astype(np.int32)

            else:

                for i in range(n_interactions):
                    density[interactions[i]] = 0

                indices[user] = np.random.choice(n_items, samples, replace=False, p=density / np.sum(density))

                for i in range(n_interactions):
                    density[interactions[i]] = density_backup[interactions[i]]

        else:
            indices[user] = np.empty(0, dtype=np.int32)

        indptr[user+1] = indptr[user] + samples
        ubatch_idx += 1

    indices = np.concatenate(indices, axis=None)
    URM_test_negative = sps.csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr),
                                       shape=URM_test.shape, dtype=np.int8)
    URM_test_negative.sort_indices()

    return URM_test_negative
