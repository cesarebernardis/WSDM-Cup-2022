#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL

from RecSysFramework.Utils import tagFilterAndStemming
from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs

from RecSysFramework.DataManager import Dataset


class Movielens20MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens20M/"
    ML_VERSION = "ml-20m"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]


    def __init__(self, reload_from_original_data=False):
        super(Movielens20MReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        zipname = "{}.zip".format(self.ML_VERSION)

        try:

            dataFile = zipfile.ZipFile(zipFile_path + zipname)

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, zipFile_path, zipname)
            dataFile = zipfile.ZipFile(zipFile_path + zipname)

        genres_path = dataFile.extract("{}/movies.csv".format(self.ML_VERSION), path=zipFile_path + "decompressed/")
        tags_path = dataFile.extract("{}/tags.csv".format(self.ML_VERSION), path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("{}/ratings.csv".format(self.ML_VERSION), path=zipFile_path + "decompressed/")

        self._print("Loading genres")
        ICM_genres, genres_mapper, item_mapper = self._loadICM_genres(genres_path, header=True, separator=',', genresSeparator="|")

        self._print("Loading tags")
        ICM_tags, tags_mapper, _ = self._loadICM_tags(tags_path, item_mapper, header=True, separator=',', if_new_item="ignore")

        self._print("Loading URM")
        URM_all, _, user_mapper = self._loadURM(URM_path, item_mapper, separator=",", header=True, if_new_user="add", if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_genres, ICM_tags, genres_mapper, tags_mapper)

        import shutil
        self._print("Cleaning temporary files")
        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all}, URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_genres": ICM_genres, "ICM_tags": ICM_tags, "ICM_all": ICM_all},
                       ICM_mappers_dict={"ICM_genres": (item_mapper.copy(), genres_mapper.copy()),
                                         "ICM_tags": (item_mapper.copy(), tags_mapper.copy()),
                                         "ICM_all": (item_mapper.copy(), feature_mapper.copy())})


    def _loadURM(self, filePath, item_mapper, header=False, separator="::", if_new_user="add", if_new_item="ignore"):

        from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_mapper, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                self._print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)
                line[-1] = line[-1].replace("\n", "")

            user_id = line[0]
            item_id = line[1]

            try:
                value = float(line[2])

                if value != 0.0:

                    URM_builder.add_data_lists([user_id], [item_id], [value])

            except:
                pass

        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


    def _loadICM_genres(self, genres_path, header=True, separator=',', genresSeparator="|"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=None, on_new_row="add")

        fileHandle = open(genres_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                self._print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)
                line[-1] = line[-1].replace("\n", "")

                movie_id = line[0]

                title = line[1]
                # In case the title contains commas, it is enclosed in "..."
                # genre list will always be the last element
                genreList = line[-1]

                genreList = genreList.split(genresSeparator)

                # Rows movie ID
                # Cols features
                ICM_builder.add_single_row(movie_id, genreList, data = 1.0)

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


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
                self._print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # If a movie has no genre, ignore it
                movie_id = line[1]

                tagList = line[2]

                # Remove non alphabetical character and split on spaces
                tagList = tagFilterAndStemming(tagList)

                # Rows movie ID
                # Cols features
                ICM_builder.add_single_row(movie_id, tagList, data = 1.0)

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

