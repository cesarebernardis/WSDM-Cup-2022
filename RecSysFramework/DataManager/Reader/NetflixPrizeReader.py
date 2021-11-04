#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Maurizio Ferrari Dacrema
"""


import zipfile, os


from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager import Dataset



class NetflixPrizeReader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EdX5jX9JDZ9JrOVps8juXvgBoUinNv_NseHmD3hMcYfsRA?e=uHyaUl"
    DATASET_SUBFOLDER = "NetflixPrize/"


    def __init__(self, reload_from_original_data=False):
        super(NetflixPrizeReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        self.zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        self.decompressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            self.dataFile = zipfile.ZipFile(self.zip_file_folder + "netflix-prize-data.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("NetflixPrizeReader: Unable to find data zip file.")
            print("NetflixPrizeReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(self.zip_file_folder))
            print("NetflixPrizeReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(self.zip_file_folder):
                os.makedirs(self.zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        URM_all, item_mapper, user_mapper = self._loadURM()

        print("NetflixPrizeReader: loading complete")

        return Dataset(self.get_dataset_name(), URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})


    def _loadURM(self):

        from RecSysFramework.Utils import IncrementalSparseMatrix

        numCells = 0
        URM_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)


        for current_split in [1, 2, 3, 4]:

            current_split_path = self.dataFile.extract("combined_data_{}.txt".format(current_split), path=self.decompressed_zip_file_folder + "decompressed/")
            fileHandle = open(current_split_path, "r")

            print("NetflixPrizeReader: loading split {}".format(current_split))

            currentMovie_id = None

            for line in fileHandle:

                if numCells % 1000000 == 0 and numCells!=0:
                    print("Processed {} cells".format(numCells))

                if (len(line)) > 1:

                    line_split = line.split(",")

                    # If line has 3 components, it is a 'user_id,rating,date' row
                    if len(line_split) == 3 and currentMovie_id!= None:

                        user_id = line_split[0]
                        URM_builder.add_data_lists([user_id], [currentMovie_id], [float(line_split[1])])
                        numCells += 1

                    # If line has 1 component, it MIGHT be a 'item_id:' row
                    elif len(line_split) == 1:
                        line_split = line.split(":")

                        # Confirm it is a 'item_id:' row
                        if len(line_split) == 2:
                            currentMovie_id = line_split[0]
                        else:
                            print("Unexpected row: '{}'".format(line))

                    else:
                        print("Unexpected row: '{}'".format(line))

            fileHandle.close()

            print("NetflixPrizeReader: cleaning temporary files")

            import shutil

            shutil.rmtree(self.decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()

