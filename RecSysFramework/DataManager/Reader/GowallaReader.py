#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import gzip, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder
import numpy as np

from RecSysFramework.DataManager import Dataset

class GowallaReader(DataReader):

    DATASET_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    DATASET_SUBFOLDER = "Gowalla/"


    ZIP_NAME = "loc-gowalla_totalCheckins.txt.gz"
    FILE_RATINGS_PATH = "loc-gowalla_totalCheckins.txt"



    def __init__(self, reload_from_original_data=False):
        super(GowallaReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            compressed_file = gzip.open(folder_path + self.ZIP_NAME, )

        except FileNotFoundError:

            self._print("Unable to find data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, folder_path, self.ZIP_NAME)

            compressed_file = gzip.open(folder_path + self.ZIP_NAME)


        URM_path = folder_path + self.FILE_RATINGS_PATH

        decompressed_file = open(URM_path, "w")

        self._save_GZ_in_text_file(compressed_file, decompressed_file)

        decompressed_file.close()

        self._print("loading URM")
        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path,
                                                                                                    header = False,
                                                                                                    separator="\t",
                                                                                                    remove_duplicates=True,
                                                                                                    custom_user_item_rating_columns = [0, 4, 2])

        # URM_all contains the coordinates in textual format
        URM_all.data = np.ones_like(URM_all.data)


        self._print("cleaning temporary files")

        os.remove(URM_path)

        self._print("loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_original_ID_to_index.copy(), item_original_ID_to_index.copy())})



    def _save_GZ_in_text_file(self, compressed_file, decompressed_file):

        print("GowallaReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("GowallaReader: decompressing file... done!")

