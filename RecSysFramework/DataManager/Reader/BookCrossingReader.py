#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, removeFeatures

from RecSysFramework.Utils import tagFilterAndStemming
from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs

from RecSysFramework.DataManager import Dataset


class BookCrossingReader(DataReader):
    """
    Collected from: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

    """

    DATASET_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    DATASET_SUBFOLDER = "BookCrossing/"
    AVAILABLE_ICM = ["ICM_all"]

    MANDATORY_POSTPROCESSINGS = []


    def __init__(self, reload_from_original_data=False):
        super(BookCrossingReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        print("BookCrossingReader: Ratings are in range 1-10, value -1 refers to an implicit rating")
        print("BookCrossingReader: ICM contains the author, publisher, year and tokens from the title")

        print("BookCrossingReader: Loading original data")

        folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("BookCrossingReader: Unable to find or extract data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, folder_path, "BX-CSV-Dump.zip")

            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")

        URM_path = dataFile.extract("BX-Book-Ratings.csv", path=folder_path + "decompressed")
        ICM_path = dataFile.extract("BX-Books.csv", path=folder_path + "decompressed")

        print("BookCrossingReader: loading ICM")
        ICM_all, feature_mapper, item_mapper = self._loadICM(ICM_path, separator=';', header=True, if_new_item="add")

        ICM_all, _, feature_mapper = removeFeatures(ICM_all, minOccurrence=5, maxPercOccurrence=0.30, reconcile_mapper=feature_mapper)

        print("BookCrossingReader: loading URM")
        URM_all, _, user_mapper = self._loadURM(URM_path, item_mapper, separator=";", header=True, if_new_user="add", if_new_item="ignore")

        print("BookCrossingReader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("BookCrossingReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all}, URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_all": ICM_all}, ICM_mappers_dict={"ICM_all": (item_mapper.copy(), feature_mapper.copy())})


    def _loadURM(self, filePath, item_mapper, header=True, separator="::", if_new_user="add", if_new_item="ignore"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_mapper, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = open(filePath, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:

            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            line = line.replace('"', '')

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                user_id = line[0]
                item_id = line[1]

                # If 0 rating is implicit
                rating = float(line[2])

                # To avoid removing it with sparse representation, set it to the sufficient rating
                if rating == 0:
                    rating = 6.0

                URM_builder.add_data_lists([user_id], [item_id], [rating])

        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


    def _loadICM(self, ICM_path, header=True, separator=',', if_new_item="add"):

        # Pubblication Data and word in title
        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=None, on_new_row=if_new_item)

        fileHandle = open(ICM_path, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:

                line = line.replace('"', '')
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                item_id = line[0]


                # Book Title
                featureTokenList = tagFilterAndStemming(line[1])
                # # Book author
                # featureTokenList.extend(tagFilterAndStemming(line[2]))
                # # Book year
                # featureTokenList.extend(tagFilterAndStemming(line[3]))
                # # Book publisher
                # featureTokenList.extend(tagFilterAndStemming(line[4]))

                #featureTokenList = tagFilterAndStemming(" ".join([line[1], line[2], line[3], line[4]]))

                featureTokenList.extend([line[2], line[3], line[4]])

                ICM_builder.add_single_row(item_id, featureTokenList, data=1.0)

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

