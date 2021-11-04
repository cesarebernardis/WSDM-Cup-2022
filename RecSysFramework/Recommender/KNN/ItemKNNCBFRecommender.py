#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import similaripy as sim

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Utils.FeatureWeighting import okapi_BM_25, TF_IDF


class ItemKNNCBF(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender):

    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBF"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, ICM_train):
        super(ItemKNNCBF, self).__init__(URM_train, ICM_train)


    def fit(self, topK=50, shrink=100, similarity='cosine', feature_weighting="none", **similarity_args):

        # Similaripy returns also self similarity, which will be set to 0 afterwards
        topK += 1
        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'"
                             .format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif feature_weighting == "TF-IDF":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)

        if similarity == "cosine":
            self.W_sparse = sim.cosine(self.ICM_train, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "jaccard":
            self.W_sparse = sim.jaccard(self.ICM_train, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "dice":
            self.W_sparse = sim.dice(self.ICM_train, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "tversky":
            self.W_sparse = sim.tversky(self.ICM_train, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "splus":
            self.W_sparse = sim.s_plus(self.ICM_train, k=topK, shrink=shrink, **similarity_args)
        else:
            raise ValueError("Unknown value '{}' for similarity".format(similarity))

        self.W_sparse.setdiag(0)
        self.W_sparse = self.W_sparse.transpose().tocsr()
