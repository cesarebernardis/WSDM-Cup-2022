#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import bz2

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder

from RecSysFramework.DataManager import Dataset


class EpinionsReader(DataReader):

    DATASET_URL = "http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
    DATASET_SUBFOLDER = "Epinions/"


    def __init__(self, reload_from_original_data=False):
        super(EpinionsReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("EpinionsReader: Loading original data")

        folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        compressed_file_path = folder_path + "ratings_data.txt.bz2"
        decompressed_file_path = folder_path + "ratings_data.txt"

        try:

            open(decompressed_file_path, "r")

        except FileNotFoundError:

            print("EpinionsReader: Unable to find decompressed data file. Decompressing...")

            try:

                compressed_file = bz2.open(compressed_file_path, "rb")

            except Exception:

                print("EpinionsReader: Unable to find or open compressed data file. Downloading...")
                downloadFromURL(self.DATASET_URL, folder_path, "ratings_data.txt.bz2")
                compressed_file = bz2.open(compressed_file_path, "rb")

            decompressed_file = open(decompressed_file_path, "w")
            self._save_BZ2_in_text_file(compressed_file, decompressed_file)
            decompressed_file.close()

        print("EpinionsReader: loading URM")

        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(decompressed_file_path, separator=" ", header=True, remove_duplicates=True)

        print("EpinionsReader: cleaning temporary files")

        import os

        os.remove(decompressed_file_path)

        print("EpinionsReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})


    def _save_BZ2_in_text_file(self, compressed_file, decompressed_file):

        print("EpinionsReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("EpinionsReader: decompressing file... done!")

