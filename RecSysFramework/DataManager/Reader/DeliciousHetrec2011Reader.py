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


class DeliciousHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip"
    DATASET_SUBFOLDER = "DeliciousHetrec2011/"


    def __init__(self, reload_from_original_data=False):
        super(DeliciousHetrec2011Reader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("DeliciousHetrec2011Reader: Loading original data")

        folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("DeliciousHetrec2011Reader: Unable to find or extract data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, folder_path, "hetrec2011-delicious-2k.zip")
            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")

        URM_path = dataFile.extract("user_taggedbookmarks-timestamps.dat", path=folder_path + "decompressed")

        print("DeliciousHetrec2011Reader: loading URM")
        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=True)

        print("DeliciousHetrec2011Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("DeliciousHetrec2011Reader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})
