#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Recommender.Utils import similarityMatrixTopK



class ItemKNNSimilarityHybrid(ItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2):
        super(ItemKNNSimilarityHybrid, self).__init__(URM_train)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybrid: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')



    def fit(self, topK=100, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()
