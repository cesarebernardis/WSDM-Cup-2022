#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Maurizio Ferrari Dacrema, Cesare Bernardis
"""

import time

from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel
from RecSysFramework.Evaluation.Evaluator import get_result_string


class Recommender_k_Fold_Wrapper:

    RECOMMENDER_NAME = "Recommender_k_Fold_Wrapper"

    def __init__(self, recommender_class, recommender_input_args_list, verbose=True):

        self.n_folds = len(recommender_input_args_list)
        self.verbose = verbose
        self.recommender_class = recommender_class
        self.RECOMMENDER_NAME = self.recommender_class.RECOMMENDER_NAME + "_k_Fold_Wrapper"
        self.recommender_input_args_list = recommender_input_args_list

        self._recommender_instance_list = [None] * self.n_folds
        self._recommender_fit_time_list = [0.0] * self.n_folds

        for current_fold in range(self.n_folds):

            self._print("Building fold {} of {}".format(current_fold+1, self.n_folds))

            recommender_input_args = self.recommender_input_args_list[current_fold]

            start_time = time.time()

            recommender_instance = self.recommender_class(
                 *recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                 **recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

            train_time = time.time() - start_time

            self._recommender_instance_list[current_fold] = recommender_instance
            self._recommender_fit_time_list[current_fold] += train_time


    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))


    def get_n_folds(self):
        return self.n_folds


    def fit(self, *posargs, **kwargs):

        if "evaluator_object" in kwargs.keys():
            evaluators = kwargs["evaluator_object"]
            assert len(evaluators) == self.n_folds, "Wrong number of evaluators for early stopping provided ({}/{})".format(len(evaluators), self.n_folds)

        for current_fold in range(self.n_folds):
            self._print("Fitting fold {} of {}".format(current_fold+1, self.n_folds))

            recommender_instance = self._recommender_instance_list[current_fold]

            if "evaluator_object" in kwargs.keys():
                kwargs["evaluator_object"] = evaluators[current_fold]

            start_time = time.time()
            recommender_instance.fit(*posargs, **kwargs)

            train_time = time.time() - start_time
            self._recommender_fit_time_list[current_fold] += train_time


    def set_recommender_fold(self, recommender_instance, fold_index):
        self._recommender_instance_list[fold_index] = recommender_instance


    def get_recommender_fold(self, fold_index):
        return self._recommender_instance_list[fold_index]


    def get_recommender_fit_time_fold(self, fold_index):
        return self._recommender_fit_time_list[fold_index]


    def save_model(self, folder_path, file_name=None):

        for fold_index in range(self.n_folds):
            if file_name is None:
                actual_file_name = self._recommender_instance_list[fold_index].RECOMMENDER_NAME
            else:
                actual_file_name = file_name
            recommender_instance = self._recommender_instance_list[fold_index]
            recommender_instance.save_model(folder_path, actual_file_name + "_fold_{}".format(fold_index))


    def load_model(self, folder_path, file_name=None):

        for fold_index in range(self.n_folds):
            if file_name is None:
                actual_file_name = self._recommender_instance_list[fold_index].RECOMMENDER_NAME
            else:
                actual_file_name = file_name
            self._recommender_instance_list[fold_index].load_model(folder_path,
                                                                   actual_file_name + "_fold_{}".format(fold_index))


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        output = {
            "epochs": 0, "best_results_run": None,
            "best_results_evaluation_time": 0,
            "best_validation_metric": 0
        }

        merge_results_run = {}

        for fold_index in range(self.n_folds):

            recommender_instance = self._recommender_instance_list[fold_index]

            assert isinstance(recommender_instance, EarlyStoppingModel), \
                "{}: Recommender instance {} does not implement early stopping"\
                    .format(self.RECOMMENDER_NAME, fold_index)

            partdict = recommender_instance.get_early_stopping_final_epochs_dict()

            output["epochs"] += partdict["epochs"]
            for k in partdict["best_results_run"].keys():
                if k in merge_results_run.keys():
                    merge_results_run[k] += partdict["best_results_run"][k]
                else:
                    merge_results_run[k] = partdict["best_results_run"][k]
            output["best_results_evaluation_time"] += partdict["best_results_evaluation_time"]
            output["best_validation_metric"] += partdict["best_validation_metric"]

        output["epochs"] = int(round(output["epochs"] / self.n_folds))
        output["best_results_evaluation_time"] /= self.n_folds
        output["best_validation_metric"] /= self.n_folds
        output["best_results_run"] = dict((k, v / self.n_folds) for k, v in merge_results_run.items())

        return output



from RecSysFramework.Evaluation.Evaluator import get_result_string

class MetricHandler_k_Fold_Wrapper():

    def __init__(self):
        self.metric_handlers = []

    def add_metric_handler(self, metric_handler):
        self.metric_handlers.append(metric_handler)

    def remove_metric_handler(self, index=None):
        if index is None:
            self.metric_handlers = []
        else:
            del self.metric_handlers[index]

    def get_results_dictionary(self, **kwargs):

        results = self.metric_handlers[0].get_results_dictionary(**kwargs)
        for metric_handler in self.metric_handlers[1:]:
            new_results = metric_handler.get_results_dictionary(**kwargs)
            for cutoff in results.keys():
                for metric in results[cutoff]:
                    results[cutoff][metric] += new_results[cutoff][metric]

        for cutoff in results.keys():
            for metric in results[cutoff]:
                results[cutoff][metric] /= len(self.metric_handlers)

        return results

    def get_evaluated_users(self, index=None):
        if index is None:
            return [mh.get_evaluated_users() for mh in self.metric_handlers]
        return self.metric_handlers[index].get_evaluated_users()

    def get_evaluated_users_count(self, index=None):
        if index is None:
            return [len(mh.get_evaluated_users()) for mh in self.metric_handlers]
        return len(self.metric_handlers[index].get_evaluated_users())

    def get_results_string(self):
        return get_result_string(self.get_results_dictionary())



class Evaluator_k_Fold_Wrapper():
    """Evaluator_k_Fold_Wrapper"""

    EVALUATOR_NAME = "Evaluator_k_Fold_Wrapper"

    def __init__(self, evaluator_instance, verbose=True):

        self.evaluator_instance = evaluator_instance
        self.n_folds = len(evaluator_instance)
        self.verbose = verbose

        self._recommender_evaluation_time_list = [None] * self.n_folds
        self._recommender_evaluation_result_list = [None] * self.n_folds


    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))


    def get_n_folds(self):
        return self.n_folds


    def get_recommender_evaluation_time_fold(self, fold_index):
        return self._recommender_evaluation_time_list[fold_index]


    def get_recommender_evaluation_result_fold(self, fold_index):
        return self._recommender_evaluation_result_list[fold_index]


    def global_setup(self, URMs_test=None, ignore_users_list=None, ignore_items_list=None):
        if URMs_test is not None:
            assert len(URMs_test) == self.n_folds, \
                "{}: Wrong number of URMs to test ({} received, {} required)" \
                    .format(self.EVALUATOR_NAME, len(URMs_test), self.n_folds)
            for i, urm in enumerate(URMs_test):
                self.evaluator_instance[i].global_setup(URM_test=urm)

        if ignore_users_list is not None:
            assert len(ignore_users_list) == self.n_folds, \
                "{}: Wrong number of users' lists to ignore ({} received, {} required)" \
                    .format(self.EVALUATOR_NAME, len(ignore_users_list), self.n_folds)
            for i, ignore_users in enumerate(ignore_users_list):
                self.evaluator_instance[i].global_setup(ignore_users=ignore_users)

        if ignore_items_list is not None:
            assert len(ignore_items_list) == self.n_folds, \
                "{}: Wrong number of items' lists to ignore ({} received, {} required)" \
                    .format(self.EVALUATOR_NAME, len(ignore_items_list), self.n_folds)
            for i, ignore_items in enumerate(ignore_items_list):
                self.evaluator_instance[i].global_setup(ignore_items=ignore_items)


    def evaluateRecommender(self, recommender_object_k_fold, URMs_test=None, ignore_users_list=None,
                            ignore_items_list=None):

        assert isinstance(recommender_object_k_fold, Recommender_k_Fold_Wrapper),\
            "{}: Recommender object is not instance of Recommender_k_Fold_Wrapper"\
            .format(self.EVALUATOR_NAME)

        assert recommender_object_k_fold.get_n_folds() == self.n_folds,\
            "{}: Recommender object has '{}' folds while Evaluator has '{}'"\
            .format(self.EVALUATOR_NAME, recommender_object_k_fold.get_n_folds(), self.n_folds)

        self.global_setup(URMs_test=URMs_test, ignore_users_list=ignore_users_list, ignore_items_list=ignore_items_list)

        metric_handlers = MetricHandler_k_Fold_Wrapper()
        for current_fold in range(self.n_folds):

            self._print("Evaluating fold {} of {}".format(current_fold+1, self.n_folds))

            recommender_fold = recommender_object_k_fold.get_recommender_fold(current_fold)
            evaluator_fold = self.evaluator_instance[current_fold]

            start_time = time.time()

            metric_handler = evaluator_fold.evaluateRecommender(recommender_fold)

            self._recommender_evaluation_time_list[current_fold] = time.time() - start_time
            self._recommender_evaluation_result_list[current_fold] = metric_handler.get_results_dictionary()
            metric_handlers.add_metric_handler(metric_handler)

        return metric_handlers




from RecSysFramework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout



def get_k_fold_bayesian_search_struct_data(dataset, dataSplitter_k_fold, recommender_class, evaluator, n_folds, random_seed=42):

    recommender_constructor_data_list = []
    recommender_constructor_data_last_list = []
    URM_test_list = []

    for URM_train, URM_test in enumerate(dataSplitter_k_fold.split(dataset, random_seed=random_seed)):

        print("Processing fold {}".format(fold_index))

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={}
        )

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train

        recommender_constructor_data_list.append(recommender_input_args)
        recommender_constructor_data_last_list.append(recommender_input_args_last_test)
        URM_test_list.append(URM_test)

    #evaluator = EvaluatorHoldout([10])

    evaluator_k_Fold = Evaluator_k_Fold_Wrapper(evaluator, n_folds)
    evaluator_k_Fold.global_setup(URMs_test=URM_test_list)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[recommender_class, n_folds, recommender_constructor_data_list],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    recommender_input_args_last = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[recommender_class, n_folds, recommender_constructor_data_last_list],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    return recommender_input_args, recommender_input_args_last, evaluator_k_Fold

