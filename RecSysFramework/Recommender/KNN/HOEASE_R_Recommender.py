#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17
@author: Cesare Bernardis

Code adapted from https://github.com/hasteck/Higher_Recsys_2021

"""

from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import seconds_to_biggest_unit

from .Cython.EASE_R_Cython import EASE_RHelper

from sklearn.preprocessing import normalize

#from numpy.linalg import inv
from scipy.linalg import inv

import numpy as np
import time
import scipy.sparse as sps
import similaripy as sim

from copy import deepcopy

### functions to create the feature-pairs
def create_list_feature_pairs(XtX, threshold):
    AA = np.triu(np.abs(XtX))
    AA[np.diag_indices(AA.shape[0])] = 0.0
    ii_pairs = np.where((AA > threshold) == True)
    return ii_pairs

def create_matrix_Z(ii_pairs, X):
    MM = np.zeros((len(ii_pairs[0]), X.shape[1]), dtype=np.float32)
    idx = np.arange(MM.shape[0])
    MM[idx, ii_pairs[0]] = 1.0
    MM[idx, ii_pairs[1]] = 1.0
    CCmask = 1.0 - MM    # see Eq. 8 in the paper
    MM = sps.csc_matrix(MM.T)
    Z =  X * MM
    Z = (Z == 2.0).astype(np.float32)
    return Z, CCmask

def get_topk(B, topK, block_size=200):
    final_shape = B.shape
    n_items = final_shape[1]
    data = np.empty((n_items, topK), dtype=np.float32)
    indices = np.empty((n_items, topK), dtype=np.int32)
    B = B.T
    for start_item in range(0, n_items, block_size):
        end_item = min(n_items, start_item + block_size)
        # relevant_items_partition is block_size x cutoff
        indices[start_item:end_item, :] = B[start_item:end_item, :].argpartition(-topK, axis=-1)[:, -topK:]
        data[start_item:end_item, :] = B[start_item:end_item, :][
            np.arange(end_item - start_item), indices[start_item:end_item, :].T].T

    return sps.csc_matrix(
        (data.flatten(), indices.flatten(), np.arange(0, n_items * topK + 1, topK)),
        shape=final_shape).tocsr()


class HOEASE_R(ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    """
        HOEASE_R

    https://dl.acm.org/doi/10.1145/3460231.3474273

    @inproceedings{10.1145/3460231.3474273,
        author = {Steck, Harald and Liang, Dawen},
        title = {Negative Interactions for Improved Collaborative Filtering: Don’t Go Deeper, Go Higher},
        year = {2021},
        isbn = {9781450384582},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3460231.3474273},
        doi = {10.1145/3460231.3474273},
        booktitle = {Fifteenth ACM Conference on Recommender Systems},
        pages = {34–43},
        numpages = {10},
        keywords = {higher order interactions, recommender systems, collaborative filtering, linear models},
        location = {Amsterdam, Netherlands},
        series = {RecSys '21}
    }

    """

    RECOMMENDER_NAME = "HOEASE_R"


    def fit(self, topK_BB=None, topK_CC=None, threshold=0.1, lambdaBB=1e3, lambdaCC=1e3,
            rho=1e5, epochs=40, verbose=True, **earlystopping_kwargs):

        if topK_BB is not None and topK_BB >= self.URM_train.shape[1]:
            topK_BB = None
        if topK_CC is not None and topK_CC >= self.URM_train.shape[1]:
            topK_CC = None

        self.verbose = verbose
        self.lambdaBB = lambdaBB
        self.lambdaCC = lambdaCC
        self.rho = rho
        self.topK_BB = topK_BB
        self.topK_CC = topK_CC
        self.epochs = epochs

        if self.verbose:
            self._print("Pre-training operations... ")

        helper = EASE_RHelper(self.URM_train)
        self.XtX = helper.dot_product()
        del helper
        self.XtXdiag = deepcopy(np.diag(self.XtX))
        self.ii_diag = np.diag_indices(self.XtX.shape[0])

        self.threshold = threshold * np.max(self.XtX)
        self.ii_feature_pairs = create_list_feature_pairs(self.XtX, self.threshold)
        self.Z, self.CCmask = create_matrix_Z(self.ii_feature_pairs, self.URM_train)

        ### create the higher-order matrices
        helper = EASE_RHelper(self.Z)
        self.ZtZ = helper.dot_product()
        del helper

        helper = EASE_RHelper(self.Z, self.URM_train)
        self.ZtX = helper.dot_product()
        del helper

        self.ZtZdiag = deepcopy(np.diag(self.ZtZ))

        self.XtX[self.ii_diag] = self.XtXdiag + self.lambdaBB
        self.PP = inv(self.XtX)
        # precompute for CC
        self.ii_diag_ZZ = np.diag_indices(self.ZtZ.shape[0])
        self.ZtZ[self.ii_diag_ZZ] = self.ZtZdiag + self.lambdaCC + self.rho

        self.QQ = inv(self.ZtZ)
        del self.ZtZ
        del self.Z

        # initialize
        self.BB = np.zeros((self.QQ.shape[0], self.XtX.shape[0]), dtype=np.float32)
        self.CC = np.zeros((self.QQ.shape[0], self.XtX.shape[0]), dtype=np.float32)
        #self.DD = np.zeros((self.QQ.shape[0], self.XtX.shape[0]), dtype=np.float32)
        self.UU = np.zeros((self.QQ.shape[0], self.XtX.shape[0]), dtype=np.float32)  # is Gamma in paper

        self._train_with_early_stopping(epochs, **earlystopping_kwargs,
                                        algorithm_name=self.RECOMMENDER_NAME)

        del self.BB
        #del self.DD
        del self.CC
        del self.UU
        del self.QQ
        del self.PP

        self.BB = self.best_BB
        if self.topK_BB is not None:
            self.BB = get_topk(self.BB, self.topK_BB)

        self.CC = self.best_CC
        if self.topK_CC is not None:
            self.CC = get_topk(self.CC, self.topK_CC)


    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.best_BB = self.BB.copy()
        self.best_CC = self.CC * self.CCmask

    def _run_epoch(self, num_epoch):
        # learn BB
        self.XtX[self.ii_diag] = self.XtXdiag
        self.BB = self.PP.dot(self.XtX - self.ZtX.T.dot(self.CC))
        gamma = np.diag(self.BB) / np.diag(self.PP)
        self.BB -= self.PP * gamma
        # learn CC
        self.CC = self.QQ.dot(self.ZtX - self.ZtX.dot(self.BB) + self.rho * (self.CC * self.CCmask - self.UU))
        # Avoid DD to save memory
        # self.DD = self.CC * self.CCmask
        # self.DD = np.maximum(0.0, self.DD) # if you want to enforce non-negative parameters
        # learn UU (is Gamma in paper)
        self.UU += self.CC - self.CC * self.CCmask

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        Xtest = self.URM_train[user_id_array]
        Ztest, _ = create_matrix_Z(self.ii_feature_pairs, Xtest)

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            pred_val = Xtest.dot(self.BB[:, items_to_compute]) + Ztest.dot(self.CC[:, items_to_compute])
            item_scores[:, items_to_compute] = pred_val
        else:
            item_scores = Xtest.dot(self.BB) + Ztest.dot(self.CC)

        return item_scores

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
            "verbose": self.verbose,
            "lambdaBB": self.lambdaBB,
            "lambdaCC": self.lambdaCC,
            "rho": self.rho,
            "topK_BB": self.topK_BB,
            "topK_CC": self.topK_CC,
            "epochs": self.epochs,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print("Saving complete")
