#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import seconds_to_biggest_unit

from .Cython.EASE_R_Cython import EASE_RHelper

from sklearn.preprocessing import normalize

import numpy as np
import time
import scipy.sparse as sps

import similaripy as sim


class EASE_R(ItemSimilarityMatrixRecommender):

    """
        EASE_R

        https://arxiv.org/pdf/1905.03375.pdf

     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }

    """

    RECOMMENDER_NAME = "EASE_R"


    def __init__(self, URM_train):
        super(EASE_R, self).__init__(URM_train)

    def fit(self, topK=None, l2_norm=1e3, normalize_matrix=False, verbose=True):

        if topK is not None and topK >= self.URM_train.shape[1]:
            topK = None

        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)

        # Grahm matrix is X X^t, compute dot product
        helper = EASE_RHelper(self.URM_train)
        grahm_matrix = helper.dot_product()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])

        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)
        del grahm_matrix

        B = P / (-np.diag(P))
        del P

        B[diag_indices] = 0.0

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

        if topK is None:
            self.W_sparse = B
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

        else:
            self.W_sparse = sim.dot_product(sps.csr_matrix(B.T), sps.eye(B.shape[1]), k=topK).transpose().tocsr()



    def _compute_score_W_dense(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)

        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def load_model(self, folder_path, file_name = None):
        super(EASE_R, self).load_model(folder_path, file_name=file_name)

        if isinstance(self.W_sparse, np.ndarray):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
