#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""



import zipfile

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder

from RecSysFramework.DataManager import Dataset


class Movielens1MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens1M/"


    def __init__(self, reload_from_original_data=False):
        super(Movielens1MReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("Movielens1MReader: Loading original data")

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens1MReader: Unable to find data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-1m.zip")
            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")

        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(URM_path, separator="::", timestamp=False)

        print("Movielens1MReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens1MReader: loading complete")

        return Dataset(self.get_dataset_name(), URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})

