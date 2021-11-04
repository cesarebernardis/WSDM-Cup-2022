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



class Movielens10MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "Movielens10M/"


    def __init__(self, reload_from_original_data=False):
        super(Movielens10MReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-10m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens10MReader: Unable to fild data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-10m.zip")
            dataFile = zipfile.ZipFile(zipFile_path + "ml-10m.zip")

        URM_path = dataFile.extract("ml-10M100K/ratings.dat", path=zipFile_path + "decompressed/")

        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(URM_path, separator="::", timestamp=False)

        print("Movielens10MReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens10MReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})
