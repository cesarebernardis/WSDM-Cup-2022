#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

import sys
import numpy as np

from RecSysFramework.Recommender.MatrixFactorization import BaseMatrixFactorizationRecommender
from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel
from RecSysFramework.Utils import check_matrix


class _MatrixFactorization_Cython(BaseMatrixFactorizationRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"


    def __init__(self, URM_train, algorithm_name="MF_WARP"):
        super(_MatrixFactorization_Cython, self).__init__(URM_train)
        self.normalize = False
        self.algorithm_name = algorithm_name


    def fit(self, epochs=300, batch_size=1000,
            num_factors=10, positive_threshold_BPR=None,
            learning_rate=0.001, use_bias=True, max_trials=10,
            sgd_mode='sgd',
            negative_interactions_quota=0.0,
            dropout_quota=None,
            init_mean=0.0, init_std_dev=0.1,
            user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0,
            verbose=False, random_seed=42,
            **earlystopping_kwargs):

        self.num_factors = num_factors
        self.use_bias = use_bias
        self.max_trials = max_trials
        self.sgd_mode = sgd_mode
        self.verbose = verbose
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, \
            "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'"\
            .format(self.RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        # Import compiled module
        from .Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch

        if self.algorithm_name in ["FUNK_SVD", "ASY_SVD"]:

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                item_reg=item_reg,
                                                                bias_reg=bias_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                dropout_quota=dropout_quota,
                                                                init_std_dev=init_std_dev,
                                                                verbose=verbose,
                                                                random_seed=random_seed)

        elif self.algorithm_name in ["MF_BPR", "MF_WARP"]:

            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            if self.positive_threshold_BPR is not None:
                URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
                URM_train_positive.eliminate_zeros()

                assert URM_train_positive.nnz > 0, \
                    "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                positive_reg=positive_reg,
                                                                negative_reg=negative_reg,
                                                                batch_size=batch_size,
                                                                max_trials=max_trials,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                init_std_dev=init_std_dev,
                                                                dropout_quota=dropout_quota,
                                                                verbose=verbose,
                                                                random_seed=random_seed)
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.algorithm_name,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias


    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()


class WARPMF(_MatrixFactorization_Cython):
    """
    Subclass allowing only for MF BPR
    """

    RECOMMENDER_NAME = "WARPMF"

    def __init__(self, *pos_args, **key_args):
        super(WARPMF, self).__init__(*pos_args, algorithm_name="MF_WARP", **key_args)

    def fit(self, **key_args):
        key_args["use_bias"] = False
        key_args["negative_interactions_quota"] = 0.0
        super(WARPMF, self).fit(**key_args)


class BPRMF(_MatrixFactorization_Cython):
    """
    Subclass allowing only for MF BPR
    """

    RECOMMENDER_NAME = "BPRMF"

    def __init__(self, *pos_args, **key_args):
        super(BPRMF, self).__init__(*pos_args, algorithm_name="MF_BPR", **key_args)

    def fit(self, **key_args):
        key_args["use_bias"] = False
        key_args["negative_interactions_quota"] = 0.0
        super(BPRMF, self).fit(**key_args)



class FunkSVD(_MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD model

    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.

    """

    RECOMMENDER_NAME = "FunkSVD"

    def __init__(self, *pos_args, **key_args):
        super(FunkSVD, self).__init__(*pos_args, algorithm_name="FUNK_SVD", **key_args)


    def fit(self, **key_args):

        super(FunkSVD, self).fit(**key_args)




class AsySVD(_MatrixFactorization_Cython):
    """
    Subclas allowing only for AsymmetricSVD model

    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    """

    RECOMMENDER_NAME = "AsySVD"

    def __init__(self, *pos_args, **key_args):
        super(AsySVD, self).__init__(*pos_args, algorithm_name="ASY_SVD", **key_args)


    def fit(self, **key_args):

        if "batch_size" in key_args and key_args["batch_size"] > 1:
            print("{}: batch_size not supported for this recommender, setting to default value 1.".format(self.RECOMMENDER_NAME))

        key_args["batch_size"] = 1

        super(AsySVD, self).fit(**key_args)



    def _prepare_model_for_validation(self):
        """
        AsymmetricSVD Computes two |n_items| x |n_features| matrices of latent factors
        ITEM_factors_Y must be used to estimate user's latent factors via the items they interacted with

        :return:
        """

        self.ITEM_factors_Y = self.cythonEpoch.get_USER_factors()
        self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y)

        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.ITEM_factors_Y_best = self.ITEM_factors_Y.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias


    def _estimate_user_factors(self, ITEM_factors_Y):

        profile_length = np.ediff1d(self.URM_train.indptr)
        profile_length_sqrt = np.sqrt(profile_length)

        # Estimating the USER_factors using ITEM_factors_Y
        if self.verbose:
            print("{}: Estimating user factors... ".format(self.algorithm_name))

        USER_factors = self.URM_train.dot(ITEM_factors_Y)

        #Divide every row for the sqrt of the profile length
        for user_index in range(self.n_users):

            if profile_length_sqrt[user_index] > 0:

                USER_factors[user_index, :] /= profile_length_sqrt[user_index]

        if self.verbose:
            print("{}: Estimating user factors... done!".format(self.algorithm_name))

        return USER_factors



    def set_URM_train(self, URM_train_new, estimate_item_similarity_for_cold_users=False, **kwargs):
        """

        :param URM_train_new:
        :param estimate_item_similarity_for_cold_users: Set to TRUE if you want to estimate the USER_factors for cold users
        :param kwargs:
        :return:
        """

        assert self.URM_train.shape == URM_train_new.shape, \
            "{}: set_URM_train old and new URM train have different shapes"\
            .format(self.RECOMMENDER_NAME)

        if len(kwargs) > 0:
            self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}"
                        .format(kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        # No need to ever use a knn model
        self._cold_user_KNN_model_available = False
        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if estimate_item_similarity_for_cold_users:

            self._print("Estimating USER_factors for cold users...")

            self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y_best)

            self._print("Estimating USER_factors for cold users... done!")



class BPRMF_AFM(_MatrixFactorization_Cython):
    """

    Subclass for BPRMF with Attribute to Feature Mapping


    """

    RECOMMENDER_NAME = "BPRMF_AFM"

    def __init__(self, URM_train, ICM, **key_args):
        super(BPRMF_AFM, self).__init__(URM_train, algorithm_name="BPRMF_AFM", **key_args)
        self.ICM = check_matrix(ICM, "csr")
        self.n_features = self.ICM.shape[1]


    def fit(self, epochs=300, batch_size=128, num_factors=10, positive_threshold_BPR=None,
            learning_rate=0.01, sgd_mode='sgd', user_reg=0.0, feature_reg=0.0,
            init_mean=0.0, init_std_dev=0.1,
            stop_on_validation=False, lower_validations_allowed=None,
            validation_metric="MAP", evaluator_object=None, validation_every_n=None):

        self.num_factors = num_factors
        self.sgd_mode = sgd_mode
        self.batch_size = batch_size
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        URM_train_positive = self.URM_train.copy()
        ICM = self.ICM.copy()

        if self.positive_threshold_BPR is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, \
                "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

        items_to_keep = np.arange(self.n_items)[np.ediff1d(URM_train_positive.tocsc().indptr) > 0]
        self.features_to_keep = np.arange(self.n_features)[np.ediff1d(ICM[items_to_keep, :].tocsc().indptr) > 0]

        from .Cython.BPRMF_AFM_Cython_epoch import BPR_AFM_Cython_Epoch

        self.cythonEpoch = BPR_AFM_Cython_Epoch(URM_train_positive.tocsr(), ICM[:, self.features_to_keep],
                                                n_factors=self.num_factors,
                                                learning_rate=learning_rate,
                                                batch_size=1,
                                                sgd_mode=sgd_mode,
                                                init_mean=init_mean,
                                                init_std_dev=init_std_dev,
                                                user_reg=user_reg,
                                                feature_reg=feature_reg)

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME
                                        )

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.ICM[:, self.features_to_keep].dot(self.cythonEpoch.get_ITEM_factors())


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ICM[:, self.features_to_keep].dot(self.ITEM_factors.copy())
