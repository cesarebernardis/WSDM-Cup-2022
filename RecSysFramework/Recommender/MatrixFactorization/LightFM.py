"""
Created on 07/01/2022

@author: Cesare Bernardis
"""

import numpy as np
import multiprocessing


from RecSysFramework.Recommender.MatrixFactorization import BaseMatrixFactorizationRecommender
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel

from lightfm import LightFM


class _LightFMModel(BaseMatrixFactorizationRecommender, EarlyStoppingModel):

    def fit(self, epochs=200, num_factors=50, k=5, n=10, user_reg=1e-6, item_reg=1e-6,
            learning_schedule='adagrad', learning_rate=0.05, rho=0.95, epsilon=1e-06,
            max_sampled=10, loss="warp", **earlystopping_kwargs):
        """

        :param epochs:
        :param num_factors:
        :param reg: Regularization constant.
        :return:
        """

        self.num_factors = num_factors
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.max_sampled = max_sampled
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.rho = rho
        self.epsilon = epsilon
        self.loss = loss
        self.k = k
        self.n = n

        self.model = LightFM(no_components=self.num_factors, loss=self.loss, max_sampled=self.max_sampled,
                learning_schedule=self.learning_schedule, learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon,
                user_alpha=self.user_reg, item_alpha=self.item_reg, random_state=100)

        self.use_bias = True
        self.GLOBAL_bias = 0.

        #self._prepare_model_for_validation()
        #self._update_best_model()

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best
        self.USER_bias = self.USER_bias_best
        self.ITEM_bias = self.ITEM_bias_best


    def _prepare_model_for_validation(self):
        self.USER_factors = self.model.user_embeddings
        self.ITEM_factors = self.model.item_embeddings
        self.ITEM_bias = self.model.item_biases
        self.USER_bias = self.model.user_biases

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.USER_bias_best = self.USER_bias.copy()
        self.ITEM_bias_best = self.ITEM_bias.copy()

    def _run_epoch(self, num_epoch):
       self.model.fit_partial(self.URM_train.tocoo(), epochs=1, num_threads=multiprocessing.cpu_count())


class BPR_LightFM(_LightFMModel):

    RECOMMENDER_NAME = "BPR"

    def fit(self, epochs=200, num_factors=50, user_reg=1e-6, item_reg=1e-6, learning_schedule='adagrad',
            learning_rate=0.05, rho=0.95, epsilon=1e-06, **earlystopping_kwargs):
        super(BPR_LightFM, self).fit(loss="bpr", epochs=epochs, num_factors=num_factors,
                    learning_schedule=learning_schedule, learning_rate=learning_rate, rho=rho, epsilon=epsilon,
                    user_reg=user_reg, item_reg=item_reg, max_sampled=max_sampled, **earlystopping_kwargs)


class WARP_LightFM(_LightFMModel):

    RECOMMENDER_NAME = "WARP"

    def fit(self, epochs=200, num_factors=50, user_reg=1e-6, item_reg=1e-6, learning_schedule='adagrad',
            learning_rate=0.05, rho=0.95, epsilon=1e-06, max_sampled=10, **earlystopping_kwargs):
        super(WARP_LightFM, self).fit(loss="warp", epochs=epochs, num_factors=num_factors,
                      learning_schedule=learning_schedule, learning_rate=learning_rate, rho=rho, epsilon=epsilon,
                      user_reg=user_reg, item_reg=item_reg, max_sampled=max_sampled, **earlystopping_kwargs)

class KOSWARP_LightFM(_LightFMModel):

    RECOMMENDER_NAME = "KOS-WARP"

    def fit(self, epochs=200, num_factors=50, k=5, n=10, user_reg=1e-6, item_reg=1e-6, learning_schedule='adagrad',
            learning_rate=0.05, rho=0.95, epsilon=1e-06, max_sampled=10, **earlystopping_kwargs):
        super(KOSWARP_LightFM, self).fit(loss="warp-kos", epochs=epochs, num_factors=num_factors, k=k, n=n,
                      learning_schedule=learning_schedule, learning_rate=learning_rate, rho=rho, epsilon=epsilon,
                      user_reg=user_reg, item_reg=item_reg, max_sampled=max_sampled, **earlystopping_kwargs)


