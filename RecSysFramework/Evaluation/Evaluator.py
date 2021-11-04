#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Maurizio Ferrari Dacrema, Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps
import time, sys, copy

from enum import Enum
from RecSysFramework.Utils import seconds_to_biggest_unit
from RecSysFramework.Utils import generate_test_negative_for_user

from .Metrics import *
from .CorrectedMetrics import *
from .Cython.helper import ordered_in1d, hypergeometric_pmf_matrix


class EvaluatorMetrics(Enum):
    ROC_AUC = ROC_AUC
    PRECISION = Precision
    PRECISION_MIN_TEST_LEN = PrecisionMinTestLen
    RECALL_MIN_TEST_LEN = RecallMinTestLen
    RECALL = Recall
    MAP = MAP
    MRR = MRR
    NDCG = NDCG
    F1 = F1
    HIT_RATE = HIT_RATE
    ARHR = ARHR
    RMSE = RMSE
    NOVELTY = Novelty
    AVERAGE_POPULARITY = AveragePopularity
    DIVERSITY_MEANINTERLIST = Diversity_MeanInterList
    DIVERSITY_HERFINDAHL = Diversity_Herfindahl
    COVERAGE_ITEM = CoverageItem
    COVERAGE_ITEM_TEST = Coverage_Test_Item
    COVERAGE_USER = CoverageUser
    DIVERSITY_GINI = Gini_Diversity
    SHANNON_ENTROPY = Shannon_Entropy

    RATIO_DIVERSITY_HERFINDAHL = Ratio_Diversity_Herfindahl
    RATIO_DIVERSITY_GINI = Ratio_Diversity_Gini
    RATIO_SHANNON_ENTROPY = Ratio_Shannon_Entropy
    RATIO_AVERAGE_POPULARITY = Ratio_AveragePopularity
    RATIO_NOVELTY = Ratio_Novelty



def get_result_string(results_run, n_decimals=7):
    output_str = ""

    for cutoff in results_run.keys():

        results_run_current_cutoff = results_run[cutoff]

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_current_cutoff.keys():
            val = results_run_current_cutoff[metric]
            if isinstance(val, np.ndarray):
                val = val.mean()
            elif isinstance(val, list):
                val = np.mean(val)
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, val, n_decimals=n_decimals)

        output_str += "\n"

    return output_str


class MetricsHandler(object):

    _BASIC_METRICS = [
        EvaluatorMetrics.PRECISION,
        EvaluatorMetrics.RECALL,
        EvaluatorMetrics.MAP,
        EvaluatorMetrics.NDCG,
        EvaluatorMetrics.ARHR,
        EvaluatorMetrics.COVERAGE_ITEM_TEST,
    ]

    def __init__(self, URM_train, URM_test, cutoff_list, metrics_list,
                 ignore_users, ignore_items, URM_test_negative=None):

        super(MetricsHandler, self).__init__()

        self.n_users, self.n_items = URM_train.shape
        self.item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        self.profile_length = np.ediff1d(URM_train.tocsr().indptr)
        self.test_positive_length = np.ediff1d(URM_test.tocsr().indptr)
        if URM_test_negative is not None:
            self.test_length = self.test_positive_length + np.ediff1d(URM_test_negative.tocsr().indptr)
        else:
            self.test_length = self.n_items - self.profile_length

        self.ignore_users = ignore_users
        self.ignore_items = ignore_items

        if isinstance(cutoff_list, list):
            self.cutoff_list = list(map(int, cutoff_list))
        else:
            self.cutoff_list = [int(cutoff_list)]
        self.max_cutoff = max(self.cutoff_list)

        if metrics_list is None:
            metrics_list = self._BASIC_METRICS
        elif not isinstance(metrics_list, list):
            metrics_list = [metrics_list]

        self.metrics = self._init_metric_objects(cutoff_list, metrics_list)
        self.evaluated_users = []


    def _init_metric_objects(self, cutoff_list, metrics_list):

        metrics_objects = {}

        for cutoff in cutoff_list:
            metrics_objects[cutoff] = []
            for metric_enum in metrics_list:
                if isinstance(metric_enum, str):
                    metric_name = metric_enum.upper().replace(" ", "_")
                    metric_enum = EvaluatorMetrics[metric_name]
                metric = metric_enum.value
                if metric is CoverageUser:
                    metrics_objects[cutoff].append(CoverageUser(self.n_users, self.ignore_users))
                elif metric is Novelty:
                    metrics_objects[cutoff].append(Novelty(self.item_popularity))
                elif metric in [Diversity_MeanInterList]:
                    metrics_objects[cutoff].append(metric(self.n_items, cutoff))
                elif issubclass(metric, CumulativeMetric):
                    metrics_objects[cutoff].append(metric())
                else:
                    metrics_objects[cutoff].append(metric(self.n_items, self.ignore_items))

        return metrics_objects


    def add_user_evaluation(self, user_id, recommended_items, relevant_predicted_ratings, relevant_items,
                            relevant_items_ratings):

        self.evaluated_users.append(user_id)
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        for cutoff in self.cutoff_list:

            is_relevant_current_cutoff = is_relevant[:cutoff]
            recommended_items_current_cutoff = recommended_items[:cutoff]
            # relevant_items_ratings_current_cutoff = relevant_items_ratings[:cutoff]

            for metric in self.metrics[cutoff]:
                if isinstance(metric, NDCG):
                    metric.add_recommendations(recommended_items_current_cutoff, relevant_items, cutoff)
                    # relevance=relevant_items_ratings)
                elif isinstance(metric, RMSE):
                    metric.add_recommendations(relevant_predicted_ratings, relevant_items_ratings)
                elif isinstance(metric, CumulativeMetric):
                    metric.add_recommendations(is_relevant_current_cutoff, relevant_items)
                elif isinstance(metric, CoverageUser):
                    metric.add_recommendations(recommended_items_current_cutoff, user_id)
                elif isinstance(metric, Coverage_Test_Item):
                    metric.add_recommendations(recommended_items_current_cutoff, is_relevant_current_cutoff)
                else:
                    metric.add_recommendations(recommended_items_current_cutoff)

    def get_results_dictionary(self, per_user=False, use_metric_name=True):

        results = {}
        for cutoff in self.metrics.keys():
            results[cutoff] = {}
            for metric in self.metrics[cutoff]:
                if use_metric_name:
                    metric_name = metric.METRIC_NAME
                else:
                    metric_name = EvaluatorMetrics(type(metric)).name
                if per_user:
                    try:
                        results[cutoff][metric_name] = metric.get_metric_value_per_user()
                    except:
                        print("MetricsHandler: Metric values per user not available for {}".format(metric_name))
                        pass
                else:
                    results[cutoff][metric_name] = metric.get_metric_value()

        return results

    def get_evaluated_users(self):
        return self.evaluated_users.copy()

    def get_evaluated_users_count(self):
        return len(self.evaluated_users)

    def get_results_string(self):
        return get_result_string(self.get_results_dictionary())

    def reset(self):
        for cutoff in self.cutoff_list:
            for metric in self.metrics[cutoff]:
                metric.reset_metric()


class CLSCorrectedEvaluatorMetrics(Enum):
    PRECISION = CLSCorrectedPrecision
    RECALL = CLSCorrectedRecall
    MAP = CLSCorrectedMAP
    NDCG = CLSCorrectedNDCG


class BVCorrectedEvaluatorMetrics(Enum):
    PRECISION = BVCorrectedPrecision
    RECALL = BVCorrectedRecall
    MAP = BVCorrectedMAP
    NDCG = BVCorrectedNDCG


class CorrectedMetricsHandler(MetricsHandler):

    def __init__(self, URM_train, URM_test, cutoff_list, metrics_list,
                 ignore_users, ignore_items, URM_test_negative=None, pmf_matrix=None):

        self.pmf_matrix = pmf_matrix
        super(CorrectedMetricsHandler, self).__init__(URM_train, URM_test, cutoff_list, metrics_list,
                     ignore_users, ignore_items, URM_test_negative=URM_test_negative)


    def _init_metric_objects(self, cutoff_list, metrics_list):

        metrics_objects = {}

        for cutoff in cutoff_list:
            metrics_objects[cutoff] = []
            for metric_enum in metrics_list:
                if isinstance(metric_enum, str):
                    metric_name = metric_enum.upper().replace(" ", "_")
                    metric_enum = self._ENUMERATOR[metric_name]
                metric = metric_enum.value

                metrics_objects[cutoff].append(
                    metric(self.n_items, max(self.test_length), cutoff, pmf_matrix=self.pmf_matrix)
                )

        return metrics_objects


    def add_user_evaluation(self, user_id, recommended_items, relevant_predicted_ratings, relevant_items,
                            relevant_items_ratings):

        self.evaluated_users.append(user_id)
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        for cutoff in self.cutoff_list:
            is_relevant_current_cutoff = is_relevant[:cutoff]
            for metric in self.metrics[cutoff]:
                metric.add_recommendations(is_relevant_current_cutoff, relevant_items)


class CLSCorrectedMetricsHandler(CorrectedMetricsHandler):

    _BASIC_METRICS = [
        CLSCorrectedEvaluatorMetrics.PRECISION,
        CLSCorrectedEvaluatorMetrics.RECALL,
        CLSCorrectedEvaluatorMetrics.MAP,
        CLSCorrectedEvaluatorMetrics.NDCG,
    ]

    _ENUMERATOR = CLSCorrectedEvaluatorMetrics


class BVCorrectedMetricsHandler(CorrectedMetricsHandler):

    _BASIC_METRICS = [
        BVCorrectedEvaluatorMetrics.PRECISION,
        BVCorrectedEvaluatorMetrics.RECALL,
        BVCorrectedEvaluatorMetrics.MAP,
        BVCorrectedEvaluatorMetrics.NDCG,
    ]

    _ENUMERATOR = BVCorrectedEvaluatorMetrics


class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator"

    def __init__(self, cutoff_list, metrics_list=None, minRatingsPerUser=1, exclude_seen=True):

        super(Evaluator, self).__init__()

        if isinstance(cutoff_list, list):
            self.cutoff_list = list(map(int, cutoff_list))
        else:
            self.cutoff_list = [int(cutoff_list)]
        self.max_cutoff = max(self.cutoff_list)

        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen
        self.metrics_list = metrics_list

        self.ignore_users_ID = np.array([])
        self.ignore_items_flag = False
        self.ignore_items_ID = np.array([])


    def global_setup(self, URM_test, ignore_users=None, ignore_items=None):
        if ignore_items is not None:
            print("{}: Ignoring {} Items".format(self.EVALUATOR_NAME, len(ignore_items)))
            self.ignore_items_flag = len(ignore_items) > 0
            self.ignore_items_ID = np.array(ignore_items)

        if ignore_users is not None:
            print("{}: Ignoring {} Users".format(self.EVALUATOR_NAME, len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)

        if URM_test is not None:
            self.URM_test = sps.csr_matrix(URM_test)
            self.URM_test.sort_indices()
            self.n_users, self.n_items = URM_test.shape


    def evaluateRecommender(self, recommender_object, URM_test=None, ignore_users=None, ignore_items=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """
        self.global_setup(URM_test=URM_test, ignore_users=ignore_users, ignore_items=ignore_items)

        assert self.URM_test is not None, "{}: Test URM not given for evaluation".format(self.EVALUATOR_NAME)

        # Prune users with an insufficient number of ratings
        numRatings = np.ediff1d(self.URM_test.tocsr().indptr)
        self.usersToEvaluate = np.arange(self.n_users)[numRatings >= self.minRatingsPerUser]
        self.usersToEvaluate = np.setdiff1d(self.usersToEvaluate, self.ignore_users_ID)

    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", \
            "{}: URM_test is not CSR, this will cause errors in getting relevant items".format(self.EVALUATOR_NAME)

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", \
            "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def _print(self, message):
        print("{}: {}".format(self.EVALUATOR_NAME, message))

    def _start_timer(self):
        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

    def _update_timer(self):

        self._n_users_evaluated += 1

        if time.time() - self._start_time_print > 30 or self._n_users_evaluated == len(self.usersToEvaluate):
            elapsed_time = time.time() - self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                self._n_users_evaluated,
                100.0 * float(self._n_users_evaluated) / len(self.usersToEvaluate),
                new_time_value, new_time_unit,
                float(self._n_users_evaluated) / elapsed_time)
            )

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()


class EvaluatorHoldout(Evaluator):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldout"

    def _run_evaluation_on_selected_users(self, recommender_object, block_size=None):

        metrics_handler = MetricsHandler(recommender_object.get_URM_train(), self.URM_test,
                                         self.cutoff_list, self.metrics_list,
                                         ignore_items=self.ignore_items_ID, ignore_users=self.ignore_users_ID)

        if block_size is None:
            block_size = min(1000, int(1e8 / self.n_items))

        user_batch_start = 0

        self._start_timer()

        while user_batch_start < len(self.usersToEvaluate):

            user_batch_end = min(user_batch_start + block_size, len(self.usersToEvaluate))

            test_user_batch_array = np.array(self.usersToEvaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, all_items_predicted_ratings = recommender_object.recommend(
                test_user_batch_array,
                remove_seen_flag=self.exclude_seen,
                cutoff=self.max_cutoff,
                remove_top_pop_flag=False,
                remove_custom_items_flag=self.ignore_items_flag,
                return_scores=True)

            # Compute recommendation quality for each user in batch
            for batch_user_index, user_id in enumerate(test_user_batch_array):

                recommended_items = np.array(recommended_items_batch_list[batch_user_index])
                predicted_ratings = all_items_predicted_ratings[batch_user_index].flatten()

                # Being the URM CSR, the indices are the non-zero column indexes
                relevant_items = self.get_user_relevant_items(user_id)
                relevant_items_ratings = self.get_user_test_ratings(user_id)

                metrics_handler.add_user_evaluation(user_id, recommended_items, predicted_ratings[relevant_items],
                                                    relevant_items, relevant_items_ratings)

                self._update_timer()

        return metrics_handler

    def evaluateRecommender(self, recommender_object, URM_test=None, ignore_users=None, ignore_items=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        super(EvaluatorHoldout, self).evaluateRecommender(recommender_object, URM_test, ignore_users, ignore_items)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        metrics_handler = self._run_evaluation_on_selected_users(recommender_object)

        if metrics_handler.get_evaluated_users_count() <= 0:
            self._print("WARNING! No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        return metrics_handler


class EvaluatorNegativeItemSample(Evaluator):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorNegativeItemSample"

    def __init__(self, cutoff_list, metrics_list=None, minRatingsPerUser=1, exclude_seen=True,
                 balanced_sampling=False, metrics_correction="None"):

        super(EvaluatorNegativeItemSample, self).__init__(cutoff_list,
                                                          metrics_list=metrics_list,
                                                          minRatingsPerUser=minRatingsPerUser,
                                                          exclude_seen=exclude_seen)
        self.balanced_sampling = balanced_sampling
        self.metrics_correction = metrics_correction
        self.pmf_matrix = None
        if self.metrics_correction == "cls":
            self.mh_class = CLSCorrectedMetricsHandler
        elif self.metrics_correction == "bv":
            self.mh_class = BVCorrectedMetricsHandler
        else:
            self.mh_class = MetricsHandler
        self.metrics_handler = None


    def _get_user_specific_items_to_compute(self, user_id):

        items_to_compute = []

        for u in user_id:
            if self.balanced_sampling:
                train_interactions = self.URM_train.indices[self.URM_train.indptr[u]:self.URM_train.indptr[u+1]]
                test_interactions = self.URM_test.indices[self.URM_test.indptr[u]:self.URM_test.indptr[u+1]]
                n_interactions = len(train_interactions) + len(test_interactions)
                samples = max(1, round((self.n_items - n_interactions) / n_interactions))
                items_to_compute.append(generate_test_negative_for_user(train_interactions, test_interactions, samples,
                                                                   self.n_items, random_seed=user_id))
            else:
                start_pos = self.URM_items_to_rank.indptr[u]
                end_pos = self.URM_items_to_rank.indptr[u+1]
                items_to_compute.append(self.URM_items_to_rank.indices[start_pos:end_pos])

        return items_to_compute


    def global_setup(self, URM_test, URM_test_negative, ignore_users=None, ignore_items=None):

        super(EvaluatorNegativeItemSample, self).global_setup(URM_test, ignore_users, ignore_items)

        if URM_test_negative is not None:
            self.URM_test_negative = sps.csr_matrix(URM_test_negative)
            self.URM_test_negative.sort_indices()

            assert self.URM_test.shape == self.URM_test_negative.shape, \
                "{}: negative and positive test URMs have different shapes".format(self.EVALUATOR_NAME)

            self.URM_items_to_rank = sps.csr_matrix(self.URM_test.astype(np.bool) +
                                                    self.URM_test_negative.astype(np.bool))
            self.URM_items_to_rank.sort_indices()
            self.URM_items_to_rank.eliminate_zeros()
            self.URM_items_to_rank.data = np.ones_like(self.URM_items_to_rank.data)

        if self.metrics_correction in ["cls", "bv"] and self.URM_items_to_rank is not None:
            n_samples = max(np.ediff1d(self.URM_items_to_rank.tocsr().indptr))
            if self.pmf_matrix is None or self.pmf_matrix.shape[0] != self.n_items or self.pmf_matrix.shape[1] != n_samples:
                self.pmf_matrix = hypergeometric_pmf_matrix(self.n_items, n_samples)


    def evaluateRecommender(self, recommender_object, URM_test=None, URM_test_negative=None,
                            ignore_users=None, ignore_items=None, filter_at_recommendation=False,
                            reinstance_metric_handler=True):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        self.global_setup(URM_test=URM_test, URM_test_negative=URM_test_negative,
                          ignore_users=ignore_users, ignore_items=ignore_items)

        if not self.balanced_sampling:
            assert self.URM_test is not None, "{}: Test URM not given for evaluation".format(self.EVALUATOR_NAME)

        # Prune users with an insufficient number of ratings
        numRatings = np.ediff1d(self.URM_test.tocsr().indptr)
        self.usersToEvaluate = np.arange(self.n_users)[numRatings >= self.minRatingsPerUser]
        self.usersToEvaluate = np.setdiff1d(self.usersToEvaluate, self.ignore_users_ID)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        self.URM_train = recommender_object.get_URM_train()

        mh_args = [self.URM_train, self.URM_test, self.cutoff_list, self.metrics_list]
        mh_kwargs = {
            "URM_test_negative": self.URM_test_negative,
            "ignore_items": self.ignore_items_ID,
            "ignore_users": self.ignore_users_ID
        }

        if self.metrics_correction in ["cls", "bv"]:
            mh_kwargs["pmf_matrix"] = self.pmf_matrix

        if reinstance_metric_handler or self.metrics_handler is None:
            self.metrics_handler = self.mh_class(*mh_args, **mh_kwargs)
        else:
            self.metrics_handler.reset()

        self._run_evaluation_on_selected_users(recommender_object, filter_at_recommendation)

        if self.metrics_handler.get_evaluated_users_count() <= 0:
            self._print("WARNING! No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        return self.metrics_handler


    def _run_evaluation_on_selected_users(self, recommender_object, filter_at_recommendation, block_size=None):

        user_batch_start = 0

        self._start_timer()

        if block_size is None:
            block_size = min(1000, int(1e8 / self.n_items))

        while user_batch_start < len(self.usersToEvaluate):
            user_batch_end = min(user_batch_start + block_size, len(self.usersToEvaluate))

            test_user_batch_array = np.array(self.usersToEvaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            items_to_compute = self._get_user_specific_items_to_compute(test_user_batch_array)

            recommended_items_batch_list, all_items_predicted_ratings = recommender_object.recommend(
                test_user_batch_array,
                remove_seen_flag=self.exclude_seen,
                cutoff=self.max_cutoff,
                remove_top_pop_flag=False,
                remove_custom_items_flag=self.ignore_items_flag,
                return_scores=True,
                items_to_compute=items_to_compute,
                filter_at_recommendation=filter_at_recommendation
            )

            for batch_user_index, test_user in enumerate(test_user_batch_array):

                #itc = items_to_compute[batch_user_index]
                recommended_items = np.array(recommended_items_batch_list[batch_user_index])
                predicted_ratings = all_items_predicted_ratings[batch_user_index].flatten()

                # Being the URM CSR, the indices are the non-zero column indexes
                relevant_items = self.get_user_relevant_items(test_user)
                relevant_items_ratings = self.get_user_test_ratings(test_user)

                self.metrics_handler.add_user_evaluation(test_user, recommended_items,
                                                    predicted_ratings[relevant_items],
                                                    relevant_items, relevant_items_ratings)

                recommender_object.reset_items_to_ignore()

                self._update_timer()

