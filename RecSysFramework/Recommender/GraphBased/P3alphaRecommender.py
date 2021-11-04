#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import check_matrix

from sklearn.preprocessing import normalize
from similaripy import p3alpha



class P3alpha(ItemSimilarityMatrixRecommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alpha"

    def __init__(self, URM_train):
        super(P3alpha, self).__init__(URM_train)


    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        self.W_sparse = p3alpha(matrix1=self.URM_train.astype(np.bool).transpose(), matrix2=self.URM_train, k=self.topK,
                                alpha=alpha, shrink=0, verbose=True, format_output="csr")

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)
