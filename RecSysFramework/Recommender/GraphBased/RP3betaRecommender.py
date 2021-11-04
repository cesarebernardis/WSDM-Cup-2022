#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import check_matrix

from sklearn.preprocessing import normalize
from similaripy import rp3beta



class RP3beta(ItemSimilarityMatrixRecommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3beta"

    def __init__(self, URM_train):
        super(RP3beta, self).__init__(URM_train)


    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.beta, self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)


    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=False, normalize_similarity=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        self.W_sparse = rp3beta(matrix1=self.URM_train.astype(np.bool).transpose(), matrix2=self.URM_train, k=self.topK,
                                alpha=alpha, beta=beta, shrink=0, verbose=True, format_output="csr")

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)
