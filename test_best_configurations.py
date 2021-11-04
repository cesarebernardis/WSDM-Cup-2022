import os

import numpy as np
import scipy.sparse as sps

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation import EvaluatorHoldout


for splitter in EXPERIMENTAL_CONFIG['splits']:

    for i, dataset_config in enumerate(EXPERIMENTAL_CONFIG['datasets']):

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']

        train, test, validation = splitter.load_split(datareader, postprocessings=postprocessings)

        evaluator = EvaluatorHoldout(cutoff_list=EXPERIMENTAL_CONFIG['cutoffs'],
                                     metrics_list=EXPERIMENTAL_CONFIG['recap_metrics'])
        evaluator.global_setup(test.get_URM())

        basepath = splitter.get_complete_default_save_folder_path(datareader, postprocessings=postprocessings)

        for algorithm in EXPERIMENTAL_CONFIG['baselines']:

            recommender_name = algorithm.RECOMMENDER_NAME
            print("Running best configuration of", recommender_name)

            dataIO = DataIO(folder_path=basepath + recommender_name + os.sep)
            data_dict = dataIO.load_data(file_name=recommender_name + "_metadata")

            recommender = algorithm(train.get_URM())
            recommender.fit(**data_dict["hyperparameters_best"])

            metrics_handler = evaluator.evaluateRecommender(recommender)
            results = metrics_handler.get_results_dictionary(use_metric_name=True)
            print(metrics_handler.get_results_string())
            print()


