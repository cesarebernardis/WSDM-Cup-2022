"""
Created on 23/03/2019

@author: Cesare Bernardis
"""

from RecSysFramework.Recommender.MatrixFactorization import BaseMatrixFactorizationRecommender
from RecSysFramework.Utils import check_matrix
import numpy as np
import implicit


class IALS(BaseMatrixFactorizationRecommender):
    """

    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf

    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    """

    RECOMMENDER_NAME = "IALS"


    def fit(self, epochs=15, num_factors=50, reg=1e-2):
        """

        :param epochs:
        :param num_factors:
        :param reg: Regularization constant.
        :return:
        """

        self.num_factors = num_factors
        self.reg = reg

        model = implicit.als.AlternatingLeastSquares(factors=self.num_factors, regularization=self.reg,
                                                     iterations=epochs, random_state=7)
        model.fit(self.URM_train.transpose().tocsr())

        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors


