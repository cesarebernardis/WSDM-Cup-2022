#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from RecSysFramework.Utils import check_matrix

import numpy as np
cimport numpy as np
import time
import sys

from cpython.array cimport array, clone

from libc.math cimport exp, sqrt
from libc.stdlib cimport malloc, free, rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item



cdef class BPR_AFM_Cython_Epoch:

    cdef int n_users, n_items, n_features, n_factors
    cdef int numPositiveIteractions

    cdef float learning_rate, user_reg, feature_reg

    cdef int batch_size

    cdef int algorithm_is_funk_svd, algorithm_is_asy_svd, algorithm_is_BPR

    cdef int[:] URM_train_indices, URM_train_indptr
    cdef double[:] URM_train_data

    cdef int[:] ICM_indices, ICM_indptr
    cdef double[:] ICM_data

    cdef double[:,:] USER_factors, ITEM_factors


    # Adaptive gradient
    cdef int useAdaGrad, useRmsprop, useAdam

    cdef double [:] sgd_cache_I, sgd_cache_U
    cdef double gamma, loss

    cdef double [:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2
    cdef double [:] sgd_cache_U_momentum_1, sgd_cache_U_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    def __init__(self, URM_train, ICM, n_factors=10,
                 learning_rate=0.01, user_reg=0.0, feature_reg=0.0,
                 init_mean=0.0, init_std_dev=0.1,
                 batch_size=1, sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(BPR_AFM_Cython_Epoch, self).__init__()


        URM_train = check_matrix(URM_train, 'csr')
        URM_train.eliminate_zeros()
        URM_train.sort_indices()

        ICM = check_matrix(ICM, 'csr')
        ICM.eliminate_zeros()
        ICM.sort_indices()

        self.numPositiveIteractions = int(URM_train.nnz * 1)
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.n_features = ICM.shape[1]
        self.n_factors = n_factors

        self.URM_train_indices = URM_train.indices
        self.URM_train_data = np.array(URM_train.data, dtype=np.float64)
        self.URM_train_indptr = URM_train.indptr

        self.ICM_indices = ICM.indices
        self.ICM_data = np.array(ICM.data, dtype=np.float64)
        self.ICM_indptr = ICM.indptr

        self.loss = 0.0
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.USER_factors = np.random.normal(self.init_mean, self.init_std_dev, (self.n_users, self.n_factors)).astype(np.float64)
        self.ITEM_factors = np.random.normal(self.init_mean, self.init_std_dev, (self.n_features, self.n_factors)).astype(np.float64)

        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache_I = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True
            self.sgd_cache_I = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True
            self.sgd_cache_I_momentum_1 = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_I_momentum_2 = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)

            self.sgd_cache_U_momentum_1 = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U_momentum_2 = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2

        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop', 'adam'. Provided value was '{}'".format(
                    sgd_mode))

        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.feature_reg = feature_reg

        if batch_size!=1:
            print("MiniBatch not implemented, reverting to default value 1")
        self.batch_size = 1


    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.URM_train_indices[self.URM_train_indptr[index]:self.URM_train_indptr[index + 1]]


    def epochIteration_Cython(self):

        # Get number of available interactions
        cdef long totalNumberOfBatch = int(len(self.URM_train_data) / self.batch_size) + 1

        cdef BPR_sample sample
        cdef long u, i, j
        cdef int i_i, i_j
        cdef long index, numCurrentBatch, processed_samples_last_print, print_block_size = 500
        cdef double x_uij, sigmoid_user, sigmoid_item, tmp

        cdef int numSeenItems, feature_union_num

        cdef double cumulative_loss = 0.0

        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        cdef int[:] feature_union
        cdef double[:] i_feature_data, j_feature_data

        feature_union = np.zeros(self.n_features, dtype=np.int32)
        i_feature_data = np.zeros(self.n_features, dtype=np.float64)
        j_feature_data = np.zeros(self.n_features, dtype=np.float64)

        for numCurrentBatch in range(totalNumberOfBatch):

            # Uniform user sampling with replacement
            sample = self.sampleBPR_Cython()

            u = sample.user
            i = sample.pos_item
            j = sample.neg_item

            i_i = self.ICM_indptr[i]
            i_j = self.ICM_indptr[j]
            feature_union_num = 0
            while i_i < self.ICM_indptr[i+1] and i_j < self.ICM_indptr[j+1]:
                if self.ICM_indices[i_i] < self.ICM_indices[i_j]:
                    feature_union[feature_union_num] = self.ICM_indices[i_i]
                    i_feature_data[feature_union_num] = self.ICM_data[i_i]
                    j_feature_data[feature_union_num] = 0.
                    i_i += 1
                elif self.ICM_indices[i_j] < self.ICM_indices[i_i]:
                    feature_union[feature_union_num] = self.ICM_indices[i_j]
                    i_feature_data[feature_union_num] = 0.
                    j_feature_data[feature_union_num] = self.ICM_data[i_j]
                    i_j += 1
                else:
                    feature_union[feature_union_num] = self.ICM_indices[i_j]
                    i_feature_data[feature_union_num] = self.ICM_data[i_i]
                    j_feature_data[feature_union_num] = self.ICM_data[i_j]
                    i_i += 1
                    i_j += 1
                feature_union_num += 1

            while i_i < self.ICM_indptr[i+1]:
                feature_union[feature_union_num] = self.ICM_indices[i_i]
                i_feature_data[feature_union_num] = self.ICM_data[i_i]
                j_feature_data[feature_union_num] = 0.
                i_i += 1

            while i_j < self.ICM_indptr[j+1]:
                feature_union[feature_union_num] = self.ICM_indices[i_j]
                i_feature_data[feature_union_num] = 0.
                j_feature_data[feature_union_num] = self.ICM_data[i_j]
                i_j += 1

            x_uij = 0.0

            for index in range(self.n_factors):
                tmp = 0
                for i_i in range(feature_union_num):
                    tmp += self.ITEM_factors[feature_union[i_i], index] * (i_feature_data[i_i] - j_feature_data[i_i])
                x_uij += self.USER_factors[u, index] * tmp

            # Use gradient of log(sigm(-x_uij))
            sigmoid_item = 1 / (1 + exp(x_uij))
            sigmoid_user = sigmoid_item

            cumulative_loss += x_uij**2

            sigmoid_user = self.adaptive_gradient_user(sigmoid_user, u)

            for index in range(self.n_factors):

                tmp = 0.0
                for i_i in range(feature_union_num):

                    tmp += self.ITEM_factors[feature_union[i_i], index] * \
                           (i_feature_data[i_i] - j_feature_data[i_i])

                    self.ITEM_factors[feature_union[i_i], index] += self.learning_rate * (sigmoid_item *
                                    (i_feature_data[i_i] - j_feature_data[i_i]) * self.USER_factors[u, index] -
                                    self.feature_reg * self.ITEM_factors[feature_union[i_i], index])

                self.USER_factors[u, index] += self.learning_rate * (sigmoid_user * tmp -
                                                                     self.user_reg * self.USER_factors[u, index])

            if processed_samples_last_print >= print_block_size or numCurrentBatch == totalNumberOfBatch-1:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                samples_per_sec = numCurrentBatch/(time.time() - start_time_epoch)

                print_block_size = int(samples_per_sec*30)

                if current_time - last_print_time > 30 or numCurrentBatch == totalNumberOfBatch-1:

                    print("Processed {} ( {:.2f}% ) in {:.2f} seconds. BPR loss {:.2E}. Sample per second: {:.0f}".format(
                        numCurrentBatch*self.batch_size,
                        100.0* numCurrentBatch/totalNumberOfBatch,
                        time.time() - last_print_time,
                        cumulative_loss/(numCurrentBatch*self.batch_size + 1),
                        float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()

        return cumulative_loss


    def get_USER_factors(self):
        return np.array(self.USER_factors)


    def get_ITEM_factors(self):
        return np.array(self.ITEM_factors)


    cdef double adaptive_gradient_item(self, double gradient, long item_id):

        cdef double gradient_update

        if self.useAdaGrad:
            self.sgd_cache_I[item_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)

        elif self.useRmsprop:
            self.sgd_cache_I[item_id] = self.sgd_cache_I[item_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)

        elif self.useAdam:

            self.sgd_cache_I_momentum_1[item_id] = \
                self.sgd_cache_I_momentum_1[item_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_I_momentum_2[item_id] = \
                self.sgd_cache_I_momentum_2[item_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_I_momentum_1[item_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_I_momentum_2[item_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)

        else:

            gradient_update = gradient

        return gradient_update


    cdef double adaptive_gradient_user(self, double gradient, long user_id):

        cdef double gradient_update

        if self.useAdaGrad:
            self.sgd_cache_U[user_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)

        elif self.useRmsprop:
            self.sgd_cache_U[user_id] = self.sgd_cache_U[user_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)

        elif self.useAdam:

            self.sgd_cache_U_momentum_1[user_id] = \
                self.sgd_cache_U_momentum_1[user_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_U_momentum_2[user_id] = \
                self.sgd_cache_U_momentum_2[user_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_U_momentum_1[user_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_U_momentum_2[user_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)

        else:

            gradient_update = gradient

        return gradient_update


    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1)
        cdef long index, start_pos_seen_items, end_pos_seen_items

        cdef int negItemSelected, numSeenItems = 0

        # Skip users with no interactions or with no negative items
        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            start_pos_seen_items = self.URM_train_indptr[sample.user]
            end_pos_seen_items = self.URM_train_indptr[sample.user+1]

            numSeenItems = end_pos_seen_items - start_pos_seen_items

        index = rand() % numSeenItems

        sample.pos_item = self.URM_train_indices[start_pos_seen_items + index]

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):

            sample.neg_item = rand() % self.n_items

            index = 0
            while index < numSeenItems and self.URM_train_indices[start_pos_seen_items + index]!=sample.neg_item:
                index+=1

            if index == numSeenItems:
                negItemSelected = True

        return sample
