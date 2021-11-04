#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import zipfile, shutil

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder

from RecSysFramework.DataManager import Dataset


class FilmTrustReader(DataReader):

    DATASET_URL = "https://www.librec.net/datasets/filmtrust.zip"
    DATASET_SUBFOLDER = "FilmTrust/"


    def __init__(self, reload_from_original_data=False):
        super(FilmTrustReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("FilmTrustReader: Loading original data")


        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("FilmTrust: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "filmtrust.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")



        URM_path = dataFile.extract("ratings.txt", path=zipFile_path + "decompressed/")

        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(URM_path, separator=" ", header=False, remove_duplicates=True)


        print("FilmTrustReader: cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("FilmTrustReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})

