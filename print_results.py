import os
import optuna
import lightgbm as lgbm

import numpy as np
import pandas as pd
import scipy.sparse as sps
import pickle as pkl

from sklearn.model_selection import GroupKFold

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Evaluation.Metrics import ndcg
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Utils import load_compressed_csr_matrix, save_compressed_csr_matrix

from utils import create_dataset_from_folder, compress_urm, read_ratings, output_scores, row_minmax_scaling, evaluate, first_level_ensemble, stretch_urm, make_submission



if __name__ == "__main__":

    exam_trains = {}
    exam_valids = {}
    exam_item_mappers = {}
    exam_user_mappers = {}

    for exam_folder in EXPERIMENTAL_CONFIG['test-datasets']:
        exam_train, exam_valid, urm_exam_valid_neg, urm_exam_test_neg = create_dataset_from_folder(exam_folder)
        exam_user_mapper, exam_item_mapper = exam_train.get_URM_mapper()
        exam_trains[exam_folder] = exam_train.get_URM()
        exam_valids[exam_folder] = exam_valid.get_URM()
        exam_item_mappers[exam_folder] = exam_item_mapper
        exam_user_mappers[exam_folder] = exam_user_mapper

    folders = [x for x in os.listdir(EXPERIMENTAL_CONFIG['dataset_folder'])
               if os.path.isdir(EXPERIMENTAL_CONFIG['dataset_folder'] + x) and ("t1" in x or "t2" in x)]
    results_df = []

    for folder in folders:

        print("Computing", folder)
        folder_train, folder_valid, _, _ = create_dataset_from_folder(folder)
        user_mapper, item_mapper = folder_train.get_URM_mapper()
        results_filename = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + "algo-performance.tsv"
        if not os.path.isfile(results_filename):
            with open(results_filename, "w") as file:
                print("folder,exam-folder,recommender,ndcg@10", file=file)

        baselines = [x for x in os.listdir(EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep)
                     if os.path.isdir(EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + x)]

        for exam_folder in exam_trains.keys():
            if exam_folder in folder:
                for recname in baselines:
                    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + recname + os.sep
                    try:
                        r = read_ratings(output_folder_path + exam_folder + "_valid_scores.tsv",
                                         exam_user_mappers[exam_folder], exam_item_mappers[exam_folder])
                        user_ndcg, n_evals = evaluate(r, exam_valids[exam_folder], cutoff=10)
                        result = sum(user_ndcg) / n_evals
                        with open(results_filename, "a") as file:
                            output = ",".join([folder, exam_folder, recname, "{:.10f}".format(result)])
                            print(output, file=file)
                    except Exception as e:
                        print(e)
                        continue

                for recname in ["ratings", "xgb", "lgbm"]:
                    output_folder_path = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep + \
                        "{}-ensemble-prediction-{}-valid.tsv.gz".format(recname, exam_folder)
                    try:
                        r = read_ratings(output_folder_path, exam_user_mappers[exam_folder], exam_item_mappers[exam_folder])
                        user_ndcg, n_evals = evaluate(r, exam_valids[exam_folder], cutoff=10)
                        result = sum(user_ndcg) / n_evals
                        with open(results_filename, "a") as file:
                            output = ",".join([folder, exam_folder, recname + "-ensemble", "{:.10f}".format(result)])
                            print(output, file=file)
                    except Exception as e:
                        print(e)
                        continue

        df = pd.read_csv(results_filename, sep=",") \
               .drop_duplicates(["folder", "exam-folder", "recommender"], keep="last", ignore_index=True)
        df.to_csv(results_filename, index=False)
        results_df.append(df)

    results_df = pd.concat(results_df)
    results_df.to_csv("all-results.csv", index=False)
