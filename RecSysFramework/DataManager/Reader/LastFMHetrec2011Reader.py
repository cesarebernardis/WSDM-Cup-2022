#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile


from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder

from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs

from RecSysFramework.DataManager import Dataset


class LastFMHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    DATASET_SUBFOLDER = "LastFMHetrec2011/"
    AVAILABLE_ICM = ["ICM_all"]


    def __init__(self, reload_from_original_data=False):
        super(LastFMHetrec2011Reader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("LastFMHetrec2011Reader: Loading original data")

        folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("LastFMHetrec2011Reader: Unable to find or extract data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, folder_path, "hetrec2011-lastfm-2k.zip")
            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")

        URM_path = dataFile.extract("user_artists.dat", path=folder_path + "decompressed")
        tags_path = dataFile.extract("user_taggedartists-timestamps.dat", path=folder_path + "decompressed")

        print("LastFMHetrec2011Reader: loading URM")
        URM_all, item_mapper, user_mapper = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=True)

        print("LastFMHetrec2011Reader: loading tags")
        ICM_tags, feature_mapper, _ = self._loadICM_tags(tags_path, item_mapper, header=True, separator='\t', if_new_item="ignore")

        print("LastFMHetrec2011Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("LastFMHetrec2011Reader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all}, URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_all": ICM_tags}, ICM_mappers_dict={"ICM_all": (item_mapper.copy(), feature_mapper.copy())})


    def _loadICM_tags(self, tags_path, item_mapper, header=True, separator=',', if_new_item="ignore"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)

        fileHandle = open(tags_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # If a movie has no genre, ignore it
                item_id = line[1]
                tag = line[2]

                # Rows movie ID
                # Cols features
                ICM_builder.add_data_lists([item_id], [tag], [1.0])

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


