#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana, Cesare Bernardis
"""


import numpy as np
import scipy.sparse as sps
import time, sys

from tqdm import tqdm

from sklearn.linear_model import ElasticNet
from RecSysFramework.Utils import seconds_to_biggest_unit, check_matrix
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



class SLIMElasticNet(ItemSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNet"

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha=1.0, positive_only=True, topK=100, verbose=True):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        itemlist = range(n_items)
        if verbose:
            itemlist = tqdm(itemlist)

        # fit each item's factors sequentially (not in parallel)
        for currentItem in itemlist:

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)




from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial


def create_shared_memory(a):
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return shm


@ignore_warnings(category=ConvergenceWarning)
def _partial_fit(items, topK, alpha, l1_ratio, urm_shape, positive_only=True, shm_names=None, shm_shapes=None, shm_dtypes=None):

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=positive_only,
        fit_intercept=False,
        copy_X=False,
        precompute=True,
        selection='random',
        max_iter=100,
        tol=1e-4
    )

    indptr_shm = shared_memory.SharedMemory(name=shm_names[0], create=False)
    indices_shm = shared_memory.SharedMemory(name=shm_names[1], create=False)
    data_shm = shared_memory.SharedMemory(name=shm_names[2], create=False)

    X_j = sps.csc_matrix((
            np.ndarray(shm_shapes[2], dtype=shm_dtypes[2], buffer=data_shm.buf).copy(),
            np.ndarray(shm_shapes[1], dtype=shm_dtypes[1], buffer=indices_shm.buf),
            np.ndarray(shm_shapes[0], dtype=shm_dtypes[0], buffer=indptr_shm.buf),
        ), shape=urm_shape)

    values, rows, cols = [], [], []

    for currentItem in items:

        y = X_j[:, currentItem].toarray()

        backup = X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]]
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0

        model.fit(X_j, y)

        nonzero_model_coef_index = model.sparse_coef_.indices
        nonzero_model_coef_value = model.sparse_coef_.data

        local_topK = min(len(nonzero_model_coef_value) - 1, topK)

        relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[:local_topK]
        relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        values.extend(nonzero_model_coef_value[ranking])
        rows.extend(nonzero_model_coef_index[ranking])
        cols.extend([currentItem] * len(ranking))

        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = backup

    indptr_shm.close()
    indices_shm.close()
    data_shm.close()

    return values, rows, cols


class MultiThreadSLIM_ElasticNet(SLIMElasticNet):

    def fit(self, alpha=1.0, l1_ratio=0.1, positive_only=True, topK=100, 
            verbose=True, workers=cpu_count()):

        assert l1_ratio>= 0 and l1_ratio<=1, \
            "ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]

        indptr_shm = create_shared_memory(self.URM_train.indptr)
        indices_shm = create_shared_memory(self.URM_train.indices)
        data_shm = create_shared_memory(self.URM_train.data)

        _pfit = partial(_partial_fit, topK=self.topK, alpha=self.alpha, urm_shape=self.URM_train.shape,
                        l1_ratio=self.l1_ratio, positive_only=self.positive_only,
                        shm_names=[indptr_shm.name, indices_shm.name, data_shm.name],
                        shm_shapes=[self.URM_train.indptr.shape, self.URM_train.indices.shape, self.URM_train.data.shape],
                        shm_dtypes=[self.URM_train.indptr.dtype, self.URM_train.indices.dtype, self.URM_train.data.dtype])

        with Pool(processes=self.workers) as pool:

            pool_chunksize = 4
            item_chunksize = 8

            itemchunks = np.array_split(np.arange(n_items), int(n_items / item_chunksize))
            if verbose:
                pbar = tqdm(total=n_items)

            # res contains a vector of (values, rows, cols) tuples
            values, rows, cols = [], [], []
            for values_, rows_, cols_ in pool.imap_unordered(_pfit, itemchunks, pool_chunksize):
                values.extend(values_)
                rows.extend(rows_)
                cols.extend(cols_)
                if verbose:
                    pbar.update(item_chunksize)

        indptr_shm.close()
        indices_shm.close()
        data_shm.close()

        indptr_shm.unlink()
        indices_shm.unlink()
        data_shm.unlink()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        self.URM_train = self.URM_train.tocsr()
