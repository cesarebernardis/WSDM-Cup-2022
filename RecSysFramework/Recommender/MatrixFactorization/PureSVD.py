#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Recommender.MatrixFactorization import BaseMatrixFactorizationRecommender

from sklearn.utils.extmath import randomized_svd

import scipy.sparse as sps
import similaripy as sim


class PureSVD(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVD"

    def __init__(self, URM_train):
        super(PureSVD, self).__init__(URM_train)


    def fit(self, num_factors=100, random_seed=None):

        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        self._print("Computing SVD decomposition... Done!")




class PureSVDSimilarity(ItemSimilarityMatrixRecommender):
    """ PureSVDSimilarity"""

    RECOMMENDER_NAME = "PureSVDSimilarity"

    def __init__(self, URM_train, sparse_weights=True):
        super(PureSVDSimilarity, self).__init__(URM_train)
        self.sparse_weights = sparse_weights


    def fit(self, num_factors=100, topK=100):

        from sklearn.utils.extmath import randomized_svd

        self._print(" Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      random_state=None)

        self._print(" Computing SVD decomposition... Done!")

        self.W_sparse = sim.dot_product(sps.csr_matrix(self.VT.T), sps.csr_matrix(sps.diags(self.Sigma)*self.VT),
                                        shrink=0.0, k=topK).transpose().tocsc()

        if not self.sparse_weights:
            self.W = self.W.toarray()

        self._print("Similarity matrix computed")
