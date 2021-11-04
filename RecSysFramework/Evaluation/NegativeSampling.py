import numpy as np
import scipy.sparse as sps

from RecSysFramework.Evaluation import Evaluator
from RecSysFramework.Evaluation import MetricsHandler


def generate_URM_test_negative(URM_train, URM_test, negative_samples=100, type="fixed"):

    type = type.lower()
    assert type in ["fixed", "per-positive"], "Unknown negative sampling type \"{}\"".format(type)

    URM = URM_train + URM_test

    indptr = np.zeros(URM.shape[0]+1, dtype=np.int32)
    indices = []

    for user in URM.shape[0]:
        interactions = URM.indices[URM.indptr[user]:URM.indptr[user+1]]
        negatives = []
        if URM_test.indptr[user] != URM_test.indptr[user+1]:
            if type == "fixed":
                samples = negative_samples
            else:
                samples = (URM_test.indptr[user + 1] - URM_test.indptr[user]) * negative_samples
            candidates = np.setdiff1d(np.arange(URM.shape[1]), interactions, assumq_unique=True)
            if len(candidates) < samples:
                print("User {} has not the required number of negative samples ({}/{})"
                      .format(user, len(candidates), samples))
                negatives = candidates
            else:
                negatives = np.random.choice(candidates, samples, replace=False)
        indices.append(negatives)
        indptr[user+1] = indptr[user] + len(indices[-1])

    indices = np.concatenate(indices, axis=None)
    URM_test_negative = sps.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr),
                                       shape=URM_test.shape)

    return URM_test_negative



class EvaluatorNegativeItemSample(Evaluator):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorNegativeItemSample"


    def _get_user_specific_items_to_compute(self, user_id):

        start_pos = self.URM_items_to_rank.indptr[user_id]
        end_pos = self.URM_items_to_rank.indptr[user_id+1]

        items_to_compute = self.URM_items_to_rank.indices[start_pos:end_pos]

        return items_to_compute


    def global_setup(self, URM_test, URM_test_negative, ignore_users=None, ignore_items=None):

        super(EvaluatorNegativeItemSample, self).global_setup(URM_test, ignore_users, ignore_items)

        if URM_test_negative is not None:

            assert self.URM_test.shape == self.URM_test_negative.shape, \
                "{}: negative and positive test URMs have different shapes".format(self.EVALUATOR_NAME)

            self.URM_test_negative = sps.csr_matrix(URM_test_negative)
            self.URM_items_to_rank = sps.csr_matrix(self.URM_test.copy().astype(np.bool)) + \
                                     sps.csr_matrix(URM_test_negative.copy().astype(np.bool))
            self.URM_items_to_rank.eliminate_zeros()
            self.URM_items_to_rank.data = np.ones_like(self.URM_items_to_rank.data)


    def evaluateRecommender(self, recommender_object, URM_test=None, URM_test_negative=None,
                            ignore_users=None, ignore_items=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        self.global_setup(URM_test, URM_test_negative, ignore_users=ignore_users, ignore_items=ignore_items)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        metrics_handler = self._run_evaluation_on_selected_users(recommender_object)

        if metrics_handler.get_evaluated_users_count() <= 0:
            self._print("WARNING - No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        return metrics_handler


    def _run_evaluation_on_selected_users(self, recommender_object):

        self._start_evaluation_timer()

        n_users_evaluated = 0

        metrics_handler = MetricsHandler(recommender_object.get_URM_train(), self.cutoff_list, self.metrics_list,
                                         ignore_items=self.ignore_items_ID, ignore_users=self.ignore_users_ID)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        for test_user in self.usersToEvaluate:

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)
            relevant_items_ratings = self.get_user_test_ratings(test_user)

            n_users_evaluated += 1

            items_to_compute = self._get_user_specific_items_to_compute(test_user)

            recommended_items, all_items_predicted_ratings = recommender_object.recommend(np.atleast_1d(test_user),
                                                          remove_seen_flag=self.exclude_seen,
                                                          cutoff=self.max_cutoff,
                                                          remove_top_pop_flag=False,
                                                          items_to_compute=items_to_compute,
                                                          remove_custom_items_flag=self.ignore_items_flag,
                                                          return_scores=True
                                                          )

            assert len(recommended_items) == 1, \
                "{}: recommended_items contained recommendations for {} users, expected was {}".format(
                        self.EVALUATOR_NAME, len(recommended_items), 1)

            assert all_items_predicted_ratings.shape[0] == 1, \
                "{}: all_items_predicted_ratings contained scores for {} users, expected was {}".format(
                        self.EVALUATOR_NAME, all_items_predicted_ratings.shape[0], 1)

            assert all_items_predicted_ratings.shape[1] == self.n_items, \
                "{}: all_items_predicted_ratings contained scores for {} items, expected was {}".format(
                        self.EVALUATOR_NAME, all_items_predicted_ratings.shape[1], self.n_items)

            recommended_items = np.array(recommended_items[0])
            recommender_object.reset_items_to_ignore()

            metrics_handler.add_user_evaluation(test_user, recommended_items, all_items_predicted_ratings.flatten(),
                                                relevant_items, relevant_items_ratings)

            self._update_evaluation_timer(n_users_evaluated)

        return metrics_handler
