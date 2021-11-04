"""
Created on 10/2017

"""


from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils.FeatureWeighting import TF_IDF, okapi_BM_25
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender

import similaripy as sim
import numpy as np
import os, sys, time
import scipy.sparse as sps



class CFW_DVV(ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "CFW_DVV"



    def __init__(self, URM_train, ICM, S_matrix_target):

        super(CFW_DVV, self).__init__(URM_train)
        self.ICM = check_matrix(ICM, 'csr')
        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.n_features = self.ICM.shape[1]
        self.sparse_weights = True


    def fit(self, n_factors=1, learning_rate=0.01, l2_reg_D=0.0, l2_reg_V=0.0, epochs=50, topK=300,
            positive_only_weights=True, precompute_common_features=True, add_zeros_quota=0.0,
            init_type="zero", sgd_mode='adagrad', gamma = 0.9, beta_1=0.9, beta_2=0.999,
            stop_on_validation=False, lower_validations_allowed=None, validation_metric="MAP",
            evaluator_object=None, validation_every_n=None):

        from .Cython.CFW_DVV_Similarity_Cython_SGD import CFW_DVV_Similarity_Cython_SGD

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.precompute_common_features = precompute_common_features
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V
        self.epochs = epochs
        self.topK = topK

        self.generate_train_data()

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

        # For mean_init use Xavier Initialization
        if self.n_factors != 0:
            std_init = 1/self.n_features/self.n_factors
        else:
            std_init = 0

        mean_init = 0

        self.cythonEpoch = CFW_DVV_Similarity_Cython_SGD(self.row_list, self.col_list, self.data_list,
                                                         self.ICM, n_factors=self.n_factors,
                                                         non_negative_weights=positive_only_weights,
                                                         precompute_common_features=precompute_common_features,
                                                         weights_initialization=weights_initialization,
                                                         learning_rate=self.learning_rate,
                                                         add_zeros_quota=add_zeros_quota,
                                                         l2_reg_D=self.l2_reg_D, l2_reg_V=self.l2_reg_V,
                                                         sgd_mode=sgd_mode, gamma=gamma,
                                                         beta_1=beta_1, beta_2=beta_2,
                                                         mean_init=mean_init, std_init=std_init)

        self.D = self.cythonEpoch.get_D()
        self.D_best = self.D.copy()

        self.V = self.cythonEpoch.get_V()
        self.V_best = self.V.copy()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME)

        self.D = self.D_best
        self.V = self.V_best

        self.compute_W_sparse()

        sys.stdout.flush()


    def set_ICM_and_recompute_W(self, ICM_new, recompute_w=True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse()


    def generate_train_data(self):

        self._print("Generating train data...")

        self.S_matrix_target.eliminate_zeros()

        numSamples = self.S_matrix_target.nnz

        zeros_to_add = int(numSamples*self.add_zeros_quota)

        self.S_matrix_target = self.S_matrix_target.tocoo()

        if zeros_to_add != 0.0:
            self.row_list = np.concatenate((np.array(self.S_matrix_target.row, dtype=np.int32), np.zeros(zeros_to_add, dtype=np.int32)))
            self.col_list = np.concatenate((np.array(self.S_matrix_target.col, dtype=np.int32), np.zeros(zeros_to_add, dtype=np.int32)))
            self.data_list = np.concatenate((np.array(self.S_matrix_target.data, dtype=np.float64), np.zeros(zeros_to_add, dtype=np.float64)))

        else:
            self.row_list = np.array(self.S_matrix_target.row, dtype=np.int32)
            self.col_list = np.array(self.S_matrix_target.col, dtype=np.int32)
            self.data_list = np.array(self.S_matrix_target.data, dtype=np.float64)

        #self.add_zeros_in_train_data_memory_intensive()
        self.add_zeros_in_train_data_row_wise()

        # Initialize samples
        self.n_samples = len(self.data_list)


    def add_zeros_in_train_data_row_wise(self):
        """
        This function uses a set of tuples to ensure the zero elements to be added are not already existent
        it is very memory intensive for big dataset
        :return:
        """

        self._print("Adding zeros in train data...")

        self.S_matrix_target = check_matrix(self.S_matrix_target, "csr")

        numSamples = self.S_matrix_target.nnz
        n_items = self.S_matrix_target.shape[0]


        zeros_to_add_global = int(numSamples * self.add_zeros_quota)
        zeros_added_global = 0

        if zeros_to_add_global + numSamples >= n_items**2:
            raise ValueError(self.RECOMMENDER_NAME + ": Too many zeros to add, not enough unique coordinates in matrix")

        zeros_to_add_per_item = int(zeros_to_add_global/self.n_items)

        while zeros_added_global < zeros_to_add_global:

            for current_item_row in range(self.n_items):

                start_pos = self.S_matrix_target.indptr[current_item_row]
                end_pos = self.S_matrix_target.indptr[current_item_row +1]

                nonzero_coordinates = set(self.S_matrix_target.indices[start_pos:end_pos])
                zeros_added_per_item = 0

                while zeros_added_per_item < zeros_to_add_per_item and zeros_added_global < zeros_to_add_global:

                    new_coordinate = np.random.randint(0, n_items)

                    if new_coordinate not in nonzero_coordinates:

                        nonzero_coordinates.add(new_coordinate)

                        self.row_list[numSamples + zeros_added_global] = current_item_row
                        self.col_list[numSamples + zeros_added_global] = new_coordinate
                        self.data_list[numSamples + zeros_added_global] = 0.0

                        zeros_added_per_item+=1
                        zeros_added_global += 1

        self._print("Added: {} zeros. Average per item is: {} ".format(zeros_added_global, zeros_to_add_per_item))
        self._print("Data points are {}".format(len(self.data_list)))


    def compute_W_sparse(self, use_D=True, use_V=True, use_incremental=False):

        self._print("Building similarity matrix...")

        start_time = time.time()
        start_time_print_batch = start_time

        # Diagonal
        if use_D:

            if use_incremental:
                D = self.D_incremental
            else:
                D = self.D_best

            self.W_sparse = sim.dot_product(self.ICM * sps.diags(D), self.ICM.T, k=self.topK).tocsr()
        else:
            self.W_sparse = sps.csr_matrix((self.n_items, self.n_items))

        if use_V:

            if use_incremental:
                V = self.V_incremental
            else:
                V = self.V_best

            # V * V.T
            W1 = self.ICM.dot(V.T)

            # Use array as it reduces memory requirements compared to lists
            dataBlock = 10000000

            values = np.zeros(dataBlock, dtype=np.float32)
            rows = np.zeros(dataBlock, dtype=np.int32)
            cols = np.zeros(dataBlock, dtype=np.int32)

            numCells = 0

            for numItem in range(self.n_items):

                V_weights = W1[numItem,:].dot(W1.T)
                V_weights[numItem] = 0.0

                relevant_items_partition = (-V_weights).argpartition(self.topK-1)[0:self.topK]
                relevant_items_partition_sorting = np.argsort(-V_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = V_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values_to_add = V_weights[top_k_idx][notZerosMask]
                rows_to_add = top_k_idx[notZerosMask]
                cols_to_add = np.ones(numNotZeros) * numItem

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.float32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.int32)))

                    rows[numCells] = rows_to_add[index]
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

                if time.time() - start_time_print_batch >= 30 or numItem==self.n_items-1:
                    columnPerSec = numItem / (time.time() - start_time)

                    self._print("Weighted similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min"
                        .format(numItem, numItem / self.n_items * 100, columnPerSec, (time.time() - start_time) / 60))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print_batch = time.time()

            V_weights = sps.csr_matrix(
                (values[:numCells], (rows[:numCells], cols[:numCells])),
                shape=(self.n_items, self.n_items),
                dtype=np.float32)

            self.W_sparse += V_weights

        self._print("Building similarity matrix... complete")


    def _prepare_model_for_validation(self):
        self.D = self.cythonEpoch.get_D()
        self.V = self.cythonEpoch.get_V()


    def _update_best_model(self):
        self.D_best = self.D.copy()
        self.V_best = self.V.copy()


    def _run_epoch(self, num_epoch):
       self.loss = self.cythonEpoch.fit()


    def save_model(self, folder_path, file_name=None):

        self._print("Saving model in folder '{}'".format(folder_path))

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        data_dict = {
            "D_best": self.D_best,
            "V_best": self.V_best,
            "topK": self.topK,
            "W_sparse": self.W_sparse,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")
