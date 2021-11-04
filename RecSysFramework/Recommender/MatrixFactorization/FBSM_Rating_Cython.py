"""
Created on 03/02/2018

"""


import similaripy as sim
import scipy.sparse as sps
import pickle, sys, time
import numpy as np

from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils.FeatureWeighting import TF_IDF, okapi_BM_25

from .Cython.FBSM_Cython_Epoch import FBSM_Cython_Epoch



class FBSM(ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "FBSM"

    def __init__(self, URM_train, ICM):

        super(FBSM, self).__init__(URM_train)

        self.ICM = check_matrix(ICM, 'csr')
        self.n_items_icm, self.n_features = ICM.shape


    def fit(self, topK=300, epochs=30, n_factors=2, learning_rate=1e-5, precompute_user_feature_count=False,
            l2_reg_D=0.01, l2_reg_V=0.01, sgd_mode='adam', gamma=0.9, beta_1=0.9, beta_2=0.999, init_type="zero",
            stop_on_validation=False, lower_validations_allowed=None, validation_metric="MAP",
            evaluator_object=None, validation_every_n=None):

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V
        self.topK = topK
        self.epochs = epochs

        if init_type == "random":
            weights_initialization = np.random.normal(0., 0.1, self.n_features).astype(np.float64)
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
            raise ValueError("FBSM: 'init_type' not recognized")

        self.cythonEpoch = FBSM_Cython_Epoch(self.URM_train, self.ICM, n_factors=self.n_factors, epochs=self.epochs,
                                                  precompute_user_feature_count=precompute_user_feature_count,
                                                  learning_rate=self.learning_rate, l2_reg_D=self.l2_reg_D,
                                                  weights_initialization=weights_initialization,
                                                  l2_reg_V=self.l2_reg_V, sgd_mode=sgd_mode, gamma=gamma,
                                                  beta_1=beta_1, beta_2=beta_2)

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


    def _prepare_model_for_validation(self):
        self.D = self.cythonEpoch.get_D()
        self.V = self.cythonEpoch.get_V()


    def _update_best_model(self):
        self.D_best = self.D.copy()
        self.V_best = self.V.copy()


    def _run_epoch(self, num_epoch):
       self.loss = self.cythonEpoch.epochIteration_Cython()


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict = {"D": self.D,
                     "V": self.V}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")


    def set_ICM_and_recompute_W(self, ICM_new, recompute_w=True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(use_D=True, use_V=True)


    def compute_W_sparse(self, use_D=True, use_V=True):

        self._print("Building similarity matrix...")

        start_time = time.time()
        start_time_print_batch = start_time

        # Diagonal
        if use_D:
            D = self.D_best
            self.W_sparse = sim.dot_product(self.ICM * sps.diags(D), self.ICM.T, k=self.topK).tocsr()
        else:
            self.W_sparse = sps.csr_matrix((self.n_items, self.n_items))

        if use_V:

            V = self.V_best
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
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = rows_to_add[index]
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

                if time.time() - start_time_print_batch >= 30 or numItem==self.n_items-1:
                    columnPerSec = numItem / (time.time() - start_time)

                    self._print("Weighted similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min"
                                .format(numItem, numItem / self.n_items * 100,
                                        columnPerSec, (time.time() - start_time) / 60))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print_batch = time.time()

            V_weights = sps.csr_matrix(
                (values[:numCells], (rows[:numCells], cols[:numCells])),
                shape=(self.n_items, self.n_items),
                dtype=np.float32)

            self.W_sparse += V_weights

        self._print("Building similarity matrix... complete")
