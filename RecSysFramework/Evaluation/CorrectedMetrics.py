#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Cesare Bernardis
"""


import numpy as np
import scipy.sparse as sps
import operator as op

from functools import reduce
from scipy.special import comb
from scipy.optimize import leastsq, least_squares
from scipy.stats import hypergeom

from numpy.linalg import pinv, inv, lstsq

from sklearn.isotonic import IsotonicRegression

from .Metrics import Metric, CumulativeMetric
from .Cython.helper import hypergeometric_pmf_matrix

class CorrectedCumulativeMetric(CumulativeMetric):

    METRIC_NAME = "CorrectedCumulativeMetric"

    def __init__(self, n_items, n_samples, cutoff, pmf_matrix=None):
        self.n_items = n_items
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.metric_per_user = []
        self.cumulative_metric = 0.0
        self.n_users = 0
        self.pmf_matrix = pmf_matrix
        if self.pmf_matrix is None:
            self.pmf_matrix = hypergeometric_pmf_matrix(self.n_items, self.n_samples)
        self.calculate_correction()
        super(CorrectedCumulativeMetric, self).__init__()


    def _add_user_result(self, result):
        self.metric_per_user.append(result)
        self.cumulative_metric += result
        self.n_users += 1


    def get_metric_value_per_user(self):
        assert self.n_users > 0, "{}: No users added, metric values not available".format(self.METRIC_NAME)
        return self.metric_per_user.copy()


    def get_metric_value(self):
        assert self.n_users > 0, "{}: No users added, metric value not available".format(self.METRIC_NAME)
        return self.cumulative_metric / self.n_users


    def merge_with_other(self, other_metric_object):
        assert other_metric_object is self.METRIC_NAME, \
            "{}: attempting to merge with a metric object of different type".format(self.METRIC_NAME)
        self.cumulative_metric += other_metric_object.cumulative_metric
        self.n_users += other_metric_object.n_users


    def add_recommendations(self, is_relevant, pos_items):
        result = self.correction[:len(is_relevant)][is_relevant > 0].sum()
        self._add_user_result(result)


    def reset_metric(self):
        self.metric_per_user = []
        self.cumulative_metric = 0.0
        self.n_users = 0



class CLSCorrectedCumulativeMetric(CorrectedCumulativeMetric):

    METRIC_NAME = "CLSCorrectedCumulativeMetric"

    def calculate_correction(self):

        def minimize(x, a, b):
            return np.matmul(a, x) - b

        p_r = np.sqrt(1 / self.n_items)

        # Old style, unfeasible with high number of samples (more than 100)
        #K = np.tile(np.atleast_2d(np.arange(self.n_items)).T, (1, self.n_samples))
        #k = np.tile(np.arange(self.n_samples), (self.n_items, 1))
        #N = self.n_items
        #n = self.n_samples-1
        #f1 = comb(K, k, exact=False)
        #f2 = comb(N-K, n-k, exact=False) / comb(N, n, exact=False)
        #A = np.multiply(f1, f2) * p_r

        A = self.pmf_matrix * p_r
        b = self._get_m_r() * p_r

        res = least_squares(minimize, np.zeros(self.n_samples), bounds=(0., 1.), args=(A, b))
        self.correction = np.clip(res.x.flatten(), 0., 1.)

        self.correction[self.cutoff:] = 0.
        #model = IsotonicRegression(increasing=False)
        #self.correction = model.fit_transform(np.arange(len(self.correction)) + 1, self.correction)



class CLSCorrectedPrecision(CLSCorrectedCumulativeMetric):

    METRIC_NAME = "Precision"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / self.cutoff
        return result



class CLSCorrectedMAP(CLSCorrectedCumulativeMetric):

    METRIC_NAME = "MAP"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / (np.arange(self.cutoff) + 1)
        return result


class CLSCorrectedRecall(CLSCorrectedCumulativeMetric):

    METRIC_NAME = "Recall"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1.
        return result


class CLSCorrectedNDCG(CLSCorrectedCumulativeMetric):

    METRIC_NAME = "NDCG"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / np.log2(np.arange(self.cutoff) + 2)
        return result



class BVCorrectedCumulativeMetric(CorrectedCumulativeMetric):

    def __init__(self, n_items, n_samples, cutoff, pmf_matrix=None, gamma=0.01):
        self.gamma = gamma
        super(BVCorrectedCumulativeMetric, self).__init__(n_items, n_samples, cutoff, pmf_matrix=pmf_matrix)

    def calculate_correction(self):

        p_r = 1 / self.n_items

        c = self.pmf_matrix.T.dot(p_r)
        p_r = np.sqrt(p_r)
        A = self.pmf_matrix * p_r
        b = self._get_m_r() * p_r

        res = np.matmul(
            np.linalg.inv(
                (1.0 - self.gamma) * np.matmul(A.T, A) +
                self.gamma * np.diag(c)
            ), A.T.dot(b)
        )
        self.correction = np.clip(res.flatten(), 0., 1.)

        self.correction[self.cutoff:] = 0.
        #model = IsotonicRegression(increasing=False)
        #self.correction = model.fit_transform(np.arange(len(self.correction)) + 1, self.correction)



class BVCorrectedPrecision(BVCorrectedCumulativeMetric):

    METRIC_NAME = "Precision"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / self.cutoff
        return result



class BVCorrectedMAP(BVCorrectedCumulativeMetric):

    METRIC_NAME = "MAP"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / (np.arange(self.cutoff) + 1)
        return result



class BVCorrectedRecall(BVCorrectedCumulativeMetric):

    METRIC_NAME = "Recall"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1.
        return result



class BVCorrectedNDCG(BVCorrectedCumulativeMetric):

    METRIC_NAME = "NDCG"

    def _get_m_r(self):
        result = np.zeros(self.n_items)
        result[:self.cutoff] = 1. / np.log2(np.arange(self.cutoff) + 2)
        return result

