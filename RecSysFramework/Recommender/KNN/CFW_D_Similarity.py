#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

"""

from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Utils.FeatureWeighting import TF_IDF, okapi_BM_25
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender

import similaripy as sim
import time, os, sys
import numpy as np
import scipy.sparse as sps



class CFW_D(ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "CFW_D"
    INIT_TYPE_VALUES = ["random", "one", "zero", "BM25", "TF-IDF"]

    def __init__(self, URM_train, ICM, S_matrix_target):

        super(CFW_D, self).__init__(URM_train)

        if URM_train.shape[1] != ICM.shape[0]:
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}"
                             .format(URM_train.shape[1], ICM.shape[0]))

        if S_matrix_target.shape[0] != S_matrix_target.shape[1]:
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}"
                             .format(S_matrix_target.shape[0], S_matrix_target.shape[1]))

        if S_matrix_target.shape[0] != ICM.shape[0]:
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}"
                             .format(S_matrix_target.shape[0], ICM.shape[0]))

        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.ICM = check_matrix(ICM, 'csr')
        self.n_features = self.ICM.shape[1]
        self.sparse_weights = True


    def generateTrainData_low_ram(self):

        self._print("Generating train data...")

        start_time_batch = time.time()

        S_matrix_contentKNN = sim.dot_product(self.ICM, self.ICM.T, k=self.topK).tocsr()
        S_matrix_contentKNN = check_matrix(S_matrix_contentKNN, "csr")

        self._print("Collaborative S density: {:.2E}, nonzero cells {}"
                    .format(self.S_matrix_target.nnz / self.S_matrix_target.shape[0]**2,
                            self.S_matrix_target.nnz))

        self._print("Content S density: {:.2E}, nonzero cells {}"
                    .format(S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2,
                            S_matrix_contentKNN.nnz))

        if self.normalize_similarity:

            # Compute sum of squared
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)

        num_common_coordinates = 0

        estimated_n_samples = int(S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0

        for row_index in range(self.n_items):

            start_pos_content = S_matrix_contentKNN.indptr[row_index]
            end_pos_content = S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row

            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index

                    new_data_value = self.S_matrix_target[row_index, col_index]

                    if self.normalize_similarity:
                        new_data_value *= sum_of_squared_features[row_index] * sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value

                    num_samples += 1

                elif np.random.rand() <= self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1

            if time.time() - start_time_batch > 30 or num_samples == S_matrix_contentKNN.nnz*(1 + self.add_zeros_quota):

                self._print("Generating train data. Sample {} ( {:.2f} %) "
                            .format(num_samples,
                                    num_samples / S_matrix_contentKNN.nnz * (1 + self.add_zeros_quota) * 100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        self._print("Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells"
                    .format(num_common_coordinates, S_matrix_contentKNN.nnz,
                            num_common_coordinates / S_matrix_contentKNN.nnz*100))

        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]

        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self._print("Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                    "average over all collaborative data is {:.2E}"
                    .format(data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))


    def compute_W_sparse(self, use_incremental=False):

        if use_incremental:
            D = self.D_incremental
        else:
            D = self.D_best

        self.W_sparse = sim.dot_product(self.ICM * sps.diags(D), self.ICM.T, k=self.topK).tocsr()
        self.sparse_weights = True


    def set_ICM_and_recompute_W(self, ICM_new, recompute_w=True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(use_incremental=False)


    def fit(self, logFile=None, precompute_common_features=True,
            learning_rate=0.01, positive_only_weights=True, init_type="zero", normalize_similarity=False,
            use_dropout=True, dropout_perc=0.3, l1_reg=0.0, l2_reg=0.0, epochs=50, topK=300,
            add_zeros_quota=0.0, sgd_mode='adagrad', gamma=0.9, beta_1=0.9, beta_2=0.999,
            stop_on_validation=False, lower_validations_allowed=None, validation_metric="MAP",
            evaluator_object=None, validation_every_n=None):

        if init_type not in self.INIT_TYPE_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_TYPE_VALUES, init_type))

        # Import compiled module
        from .Cython.CFW_D_Similarity_Cython_SGD import CFW_D_Similarity_Cython_SGD

        self.logFile = logFile

        self.positive_only_weights = positive_only_weights
        self.normalize_similarity = normalize_similarity
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.topK = topK

        self.generateTrainData_low_ram()

        if init_type == "random":
            weights_initialization = np.random.normal(0.001, 0.1, self.n_features).astype(np.float64)
        elif init_type == "one":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
        elif init_type == "zero":
            weights_initialization = np.zeros(self.n_features, dtype=np.float64)
        elif init_type == "BM25":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif init_type == "TF-IDF":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        else:
            raise ValueError("CFW_D_Similarity_Cython: 'init_type' not recognized")

        # Instantiate fast Cython implementation
        self.cythonEpoch = CFW_D_Similarity_Cython_SGD(self.row_list, self.col_list, self.data_list,
                                                      self.n_features, self.ICM,
                                                      precompute_common_features=precompute_common_features,
                                                      non_negative_weights=self.positive_only_weights,
                                                      weights_initialization=weights_initialization,
                                                      use_dropout=use_dropout, dropout_perc=dropout_perc,
                                                      learning_rate=learning_rate,
                                                      l1_reg=l1_reg, l2_reg=l2_reg, sgd_mode=sgd_mode,
                                                      gamma=gamma, beta_1=beta_1, beta_2=beta_2)

        self._print("Initialization completed")

        self.D = self.cythonEpoch.get_D()
        self.D_best = self.D.copy()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME)

        self.D = self.D_best
        self.compute_W_sparse()
        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.D = self.cythonEpoch.get_D()


    def _update_best_model(self):
        self.D_best = self.D.copy()


    def _run_epoch(self, num_epoch):
        self.loss = self.cythonEpoch.fit()


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict = {
            "D_best": self.D_best,
            "topK": self.topK,
            "sparse_weights": self.sparse_weights,
            "W_sparse": self.W_sparse,
            "normalize_similarity": self.normalize_similarity
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")

