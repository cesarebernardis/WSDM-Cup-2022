#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

"""

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Recommender.DataIO import DataIO

import numpy as np
import scipy.sparse as sps

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import lstsq


def tr(x, y):
    if sps.isspmatrix(x) or sps.isspmatrix(y):
        return x.multiply(y).sum()
    else:
        return np.multiply(x, y).sum()


class LCE(Recommender):

    """
    Local Collective Embeddings
    "Item Cold-Start Recommendations: Learning Local Collective Embeddings"
    Martin Saveski et al.

    """

    RECOMMENDER_NAME = "LCE"

    def __init__(self, URM_train, ICM):

        super(LCE, self).__init__(URM_train)

        # CSR is faster during evaluation
        self.n_users, self.n_items = self.URM_train.shape
        self.ICM = check_matrix(ICM, 'csr')
        self.n_features = self.ICM.shape[1]


    def construct_A(self, X, k, binary=False):
        
        """
        
        Constructs an adjacency matrix based on the nearest neighbor graph

        Parameters:
        X: data matrix where every point is a row

        k: number of nearest neighbors to be included in the graph

        binary: boolean, whether to include boolean values or the cosine
           similarity.

        """

        nbrs = NearestNeighbors(n_neighbors=1 + k, n_jobs=-1).fit(X)
        if binary:
            return nbrs.kneighbors_graph(X)
        else:
            return nbrs.kneighbors_graph(X, mode='distance')


    def fit(self, k=500, alpha=0.5, beta=0.05, lamb=0.5,
            epsilon=0.001, maxiter=500, rnd_seed=42, verbose=False):
        
        """

        Adapted from matlab implementation at:
        https://github.com/msaveski/LCE

        Parameters:

        k: number of topics. Controls the complexity of the model.

        alpha in [0, 1] controls the importance of each factorization.
         Setting alpha = 0.5 gives equal importance to both factorizations, while
         values of alpha >0.5 (or alpha < 0.5) give more importance to the
         factorization of Xs (or Xu).
         Default: 0.5

        beta: Controls the influence of the Laplacian regularization.

        lambda: hyper-paramter controlling the Thikonov Regularization.
         Enforces smooth solutions and avoids over fitting.
         Default: 0.5

        epsilon: Threshold controlling the number of iterations.
         If the objective function decreases less then epsilon from one iteration
         to another, the optimization procedure is stopped.

        maxiter: Maximum number of iterations.

        verbose: True|False. If set to True prints the value of the objective function
         in each iteration.
        
        """

        super(LCE, self).fit()

        np.random.seed(rnd_seed)

        Xs = self.ICM.copy()
        Xu = self.URM_train.transpose(copy=True)

        A = self.construct_A(Xs, 1, True)
        self._print("Matrix A contructed".format(self.RECOMMENDER_NAME))

        n = Xs.shape[0]
        v1 = Xs.shape[1]
        v2 = Xu.shape[1]

        W = np.random.uniform(0., 1., (n, k))
        Hs = np.random.uniform(0., 1., (k, v1))
        Hu = np.random.uniform(0., 1., (k, v2))

        D = sps.diags(A.sum(axis=0).A.flatten(), shape=A.shape)

        gamma = 1 - alpha
        trXstXs = tr(Xs, Xs)
        trXutXu = tr(Xu, Xu)

        Wt = W.transpose().conjugate()
        WtW = np.dot(Wt, W)
        WtXs = sps.csr_matrix.dot(Wt, Xs)
        WtXu = sps.csr_matrix.dot(Wt, Xu)
        WtWHs = np.dot(WtW, Hs)
        WtWHu = np.dot(WtW, Hu)
        DW = D.dot(W)
        AW = A.dot(W)

        itNum = 1
        ObjHistory = []
        delta = 2 * epsilon

        self._print("Preprocessing completed, starting iterations".format(self.RECOMMENDER_NAME))

        while delta > epsilon and itNum <= maxiter:

            Hs = np.multiply(Hs, np.divide(alpha * WtXs, np.maximum(alpha * WtWHs + lamb * Hs, 1e-10)))
            Hu = np.multiply(Hu, np.divide(gamma * WtXu, np.maximum(gamma * WtWHu + lamb * Hu, 1e-10)))

            W = np.multiply(W, np.divide(alpha * Xs.dot(Hs.transpose().conjugate()) +
                                         gamma * Xu.dot(Hu.transpose().conjugate()) + beta * AW,
                        np.maximum(alpha * np.dot(W, np.dot(Hs, Hs.transpose().conjugate())) +
                                   gamma * np.dot(W, np.dot(Hu, Hu.transpose().conjugate())) + beta * DW + lamb * W,
                                   1e-10)))

            Wt = W.transpose().conjugate()
            WtW = np.dot(Wt, W)
            WtXs = sps.csr_matrix.dot(Wt, Xs)
            WtXu = sps.csr_matrix.dot(Wt, Xu)
            WtWHs = np.dot(WtW, Hs)
            WtWHu = np.dot(WtW, Hu)
            DW = D.dot(W)
            AW = A.dot(W)

            tr1 = alpha * (trXstXs - 2 * tr(Hs, WtXs) + tr(Hs, WtWHs))
            tr2 = gamma * (trXutXu - 2 * tr(Hu, WtXu) + tr(Hu, WtWHu))
            tr3 = beta * (tr(W, DW) - tr(W, AW))
            tr4 = lamb * (np.trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu))
            Obj = tr1 + tr2 + tr3 + tr4
            ObjHistory.append(Obj)

            if itNum > 1:
                delta = np.abs(ObjHistory[-1] - ObjHistory[-2])
                if verbose:
                    self._print('Iteration: {} \t Objective: {:.5f} \t Delta: {:.5f}'
                                .format(self.RECOMMENDER_NAME, itNum, Obj, delta))
            elif verbose:
                self._print('Iteration: {} \t Objective: {:.5f}'
                            .format(self.RECOMMENDER_NAME, itNum, Obj))

            itNum += 1

        self.W = W
        self.Hs = Hs
        self.Hu = Hu

        self.Ws_test = []
        batch_size = 1000
        for i in range(int(self.n_items / batch_size) + 1):
            if verbose:
                self._print("Calculating Ws test ({}/{})"
                            .format(self.RECOMMENDER_NAME, i, int(self.n_items / batch_size)))
            start = i * batch_size
            end = min(self.n_items, (1+i) * batch_size)
            self.Ws_test.append(lstsq(Hs.T, Xs[start:end, :].toarray().T, rcond=None)[0].T)

        self.Ws_test = np.vstack(self.Ws_test)


    def compute_item_score(self, user_id_array, items_to_compute=None):

        assert self.Hu.shape[1] > user_id_array.max(),\
                "PureSVD: Cold users not allowed. Users in trained model are {}, " \
                "requested prediction for users up to {}".format(self.Hu.shape[1], user_id_array.max())

        if items_to_compute is not None:
            item_scores = self.Ws_test.dot(self.Hu[:, user_id_array])
        else:
            item_scores = self.Ws_test.dot(self.Hu[:, user_id_array])

        return item_scores.T


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "Ws_test": self.Ws_test,
            "Hs": self.Hs,
            "Hu": self.Hu,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")

