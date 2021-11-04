#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix

import ast
import tarfile, os


def parse_json(file_path):
    g = open(file_path, 'r', encoding="latin1")

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))



class YelpReader(DataReader):
    """
    Documentation here: https://www.yelp.com/dataset/documentation/main
    """
    #DATASET_URL = "https://www.yelp.com/dataset"
    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EXwcwdmew5dLgmbCHMXyX-4B0rGDMH1qIpZEo4OS0xCZ-w?e=knMda2"
    DATASET_SUBFOLDER = "Yelp/"


    def __init__(self, reload_from_original_data=False):
        super(YelpReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        # Load data from original

        print("YelpReader: Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            compressed_file = tarfile.open(compressed_file_folder + "yelp_dataset.tar", "r")
            compressed_file.extract("yelp_academic_dataset_review.json", path=decompressed_file_folder + "decompressed/")
            compressed_file.close()

        except (FileNotFoundError, tarfile.ReadError, tarfile.ExtractError):

            print("YelpReader: Unable to fild or decompress tar.gz file.")
            print("YelpReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_file_folder))
            print("YelpReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        URM_path = decompressed_file_folder + "decompressed/yelp_academic_dataset_review.json"

        print("YelpReader: loading URM")
        URM_all_builder = self._loadURM(URM_path, if_new_user="add", if_new_item="add")

        URM_all = URM_all_builder.get_SparseMatrix()

        item_mapper = URM_all_builder.get_column_token_to_id_mapper()
        user_mapper = URM_all_builder.get_row_token_to_id_mapper()

        print("YelpReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        print("YelpReader: loading complete")

        return Dataset(self.get_dataset_name(), URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())})


    def _loadURM (self, filePath, if_new_user="add", if_new_item="add"):

        URM_all_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        parser_metadata = parse_json(filePath)

        numMetadataParsed = 0

        for newMetadata in parser_metadata:

            numMetadataParsed+=1

            if (numMetadataParsed % 1000000 == 0):
                print("Processed {}".format(numMetadataParsed))

            business_id = newMetadata["business_id"]
            user_id = newMetadata["user_id"]
            rating = float(newMetadata["stars"])

            URM_all_builder.add_data_lists([user_id], [business_id], [rating])

        return  URM_all_builder


