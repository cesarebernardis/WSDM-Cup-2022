#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import zipfile, shutil

from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs
from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL
import numpy as np

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Recommender.Utils import reshapeSparse

class FrappeReader(DataReader):

    DATASET_URL = "https://github.com/hexiangnan/neural_factorization_machine/archive/master.zip"
    DATASET_SUBFOLDER = "Frappe/"


    def __init__(self, reload_from_original_data=False):
        super(FrappeReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "neural_factorization_machine-master.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")



        inner_path_in_zip = "neural_factorization_machine-master/data/frappe/"


        URM_train_path = dataFile.extract(inner_path_in_zip + "frappe.train.libfm", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(inner_path_in_zip + "frappe.test.libfm", path=zipFile_path + "decompressed/")
        URM_validation_path = dataFile.extract(inner_path_in_zip + "frappe.validation.libfm", path=zipFile_path + "decompressed/")


        tmp_URM_train, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_train_path,
                                                                                             item_original_ID_to_index = None,
                                                                                             user_original_ID_to_index = None)

        tmp_URM_test, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_test_path,
                                                                                             item_original_ID_to_index = item_original_ID_to_index,
                                                                                             user_original_ID_to_index = user_original_ID_to_index)

        tmp_URM_validation, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_validation_path,
                                                                                             item_original_ID_to_index = item_original_ID_to_index,
                                                                                             user_original_ID_to_index = user_original_ID_to_index)

        shape = (len(user_original_ID_to_index), len(item_original_ID_to_index))

        tmp_URM_train = reshapeSparse(tmp_URM_train, shape)
        tmp_URM_test = reshapeSparse(tmp_URM_test, shape)
        tmp_URM_validation = reshapeSparse(tmp_URM_validation, shape)


        URM_occurrence = tmp_URM_train + tmp_URM_test + tmp_URM_validation

        URM_all = URM_occurrence.copy()
        URM_all.data = np.ones_like(URM_all.data)


        print("FilmTrustReader: cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("FilmTrustReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_original_ID_to_index.copy(), item_original_ID_to_index.copy())})







    def _loadURM(self, file_name, header = False,
                 separator = " ",
                 item_original_ID_to_index = None,
                 user_original_ID_to_index = None):



        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index, on_new_col = "add",
                                                        preinitialized_row_mapper = user_original_ID_to_index, on_new_row = "add")


        fileHandle = open(file_name, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:

            if (numCells % 100000 == 0 and numCells!=0):
                print("Processed {} cells".format(numCells))

            line = line.split(separator)
            if (len(line)) > 1:
                if line[0]=='-1':
                    numCells += 1
                    continue
                elif line[0]=='1':
                    item = int(line[2].split(':')[0])
                    user = int(line[1].split(':')[0])
                    value = 1.0
                else:
                    print('ERROR READING DATASET')
                    break



            numCells += 1

            URM_builder.add_data_lists([user], [item], [value])



        fileHandle.close()


        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()
