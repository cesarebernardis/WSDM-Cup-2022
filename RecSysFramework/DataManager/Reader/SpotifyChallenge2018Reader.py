#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import pandas as pd
import zipfile, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager import Dataset

from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs


class SpotifyChallenge2018Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/Ef77B4q8faxEpflGU-kkALIBcb0HUOxo8ER1u_PZNVTPvw?e=7lAc3u"
    DATASET_SUBFOLDER = "SpotifyChallenge2018/"


    def __init__(self, reload_from_original_data=False):
        super(SpotifyChallenge2018Reader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("SpotifyChallenge2018Reader: Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(compressed_file_folder + "dataset_challenge.zip")
            URM_path = dataFile.extract("interactions.csv", path=decompressed_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("SpotifyChallenge2018Reader: Unable to fild data zip file.")
            print("SpotifyChallenge2018Reader: Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            print("SpotifyChallenge2018Reader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        URM_all, item_mapper, user_mapper = self._load_URM(URM_path, if_new_user="add", if_new_item="add")

        print("SpotifyChallenge2018Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_file_folder + "decompressed", ignore_errors=True)

        print("SpotifyChallenge2018Reader: loading complete")

        return Dataset(self.get_dataset_name(), URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})


    def _load_URM(self, URM_path, if_new_user="add", if_new_item="add"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        df = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                        usecols=['pid','tid','pos'],
                        dtype={'pid':np.int32,'tid':np.int32,'pos':np.int32})

        playlists = df['pid'].values
        tracks = df['tid'].values

        URM_builder.add_data_lists(playlists, tracks, np.ones_like(playlists))

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()




