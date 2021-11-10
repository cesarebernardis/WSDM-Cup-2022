#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Massimo Quadrana
"""

import numpy as np
from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import check_matrix



class TopPop(Recommender):
    """Top Popular recommender"""

    RECOMMENDER_NAME = "TopPop"

    def __init__(self, URM_train):
        super(TopPop, self).__init__(URM_train)


    def fit(self):

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.log10(np.ediff1d(self.URM_train.tocsc().indptr) + 1)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))



class GlobalEffects(Recommender):
    """docstring for GlobalEffects"""

    RECOMMENDER_NAME = "GlobalEffects"

    def __init__(self, URM_train):
        super(GlobalEffects, self).__init__(URM_train)


    def fit(self, lambda_user=10, lambda_item=25):

        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.n_items = self.URM_train.shape[1]


        # convert to csc matrix for faster column-wise sum
        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        # 1) global average
        self.mu = self.URM_train.data.sum(dtype=np.float32) / len(self.URM_train.data)

        # 2) item average bias
        # compute the number of non-zero elements for each column
        col_nnz = np.ediff1d(self.URM_train.indptr)

        # it is equivalent to:
        # col_nnz = X.indptr[1:] - X.indptr[:-1]
        # and it is **much faster** than
        # col_nnz = (X != 0).sum(axis=0)

        URM_train_unbiased = self.URM_train.tocsc(copy=True)
        URM_train_unbiased.data -= self.mu
        self.item_bias = np.divide(np.array(URM_train_unbiased.sum(axis=0)).flatten(), col_nnz + self.lambda_item)

        # 3) user average bias
        # NOTE: the user bias is *useless* for the sake of ranking items. We just show it here for educational purposes.

        # first subtract the item biases from each column
        # then repeat each element of the item bias vector a number of times equal to col_nnz
        # and subtract it from the data vector
        URM_train_unbiased.data -= np.repeat(self.item_bias, col_nnz)

        # now convert the csc matrix to csr for efficient row-wise computation
        URM_train_unbiased_csr = URM_train_unbiased.tocsr()
        row_nnz = np.ediff1d(URM_train_unbiased_csr.indptr)
        # finally, let's compute the bias
        self.user_bias = np.divide(np.array(URM_train_unbiased_csr.sum(axis=1)).flatten(), row_nnz + self.lambda_user)

        # 4) precompute the item ranking by using the item bias only
        # the global average and user bias won't change the ranking, so there is no need to use them
        #self.item_ranking = np.argsort(self.bi)[::-1]

        self.URM_train = check_matrix(self.URM_train, 'csr', dtype=np.float32)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            partial_item_scores = np.add.outer(self.user_bias[user_id_array], self.item_bias[items_to_compute]).astype(np.float32) + self.mu
            item_scores[:, items_to_compute] = self.mu + np.add.outer(self.user_bias[user_id_array],
                                                            self.item_bias[items_to_compute]).astype(np.float32)
        else:
            item_scores = np.add.outer(self.user_bias[user_id_array], self.item_bias).astype(np.float32) + self.mu

        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {"item_bias": self.item_bias}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))



class Random(Recommender):
    """Random recommender"""

    RECOMMENDER_NAME = "RandomRecommender"

    def __init__(self, URM_train):
        super(Random, self).__init__(URM_train)


    def fit(self, random_seed=42):
        np.random.seed(random_seed)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a random block (len(user_id_array), n_items) array with the item score

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.random.rand(len(user_id_array), len(items_to_compute))

        else:
            item_scores = np.random.rand(len(user_id_array), self.n_items)

        return item_scores


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

