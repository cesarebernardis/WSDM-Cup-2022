#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Recommender.KNN import ItemKNNCBF

import scipy.sparse as sps
import numpy as np


class ItemKNNCFCBFHybrid(ItemKNNCBF, ItemSimilarityMatrixRecommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "ItemKNNCFCBFHybrid"


    def fit(self, ICM_weight = 1.0, **fit_args):

        self.ICM_train = self.ICM_train*ICM_weight
        self.ICM_train = sps.hstack([self.ICM_train, self.URM_train.T], format='csr')

        super(ItemKNNCFCBFHybrid, self).fit(**fit_args)


    def _get_cold_item_mask(self):
        return np.logical_and(self._cold_item_CBF_mask, self._cold_item_mask)

