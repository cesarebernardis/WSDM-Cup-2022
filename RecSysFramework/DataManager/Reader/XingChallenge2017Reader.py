#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs


class XingChallenge2017Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EYtfBPdYbbxAjPc8hvi5vCgBVXEEM5s5R6zwHC0JAPGG4w?e=OORsFX"
    DATASET_SUBFOLDER = "XingChallenge2017/"
    AVAILABLE_ICM = ["ICM_all"]


    def __init__(self, reload_from_original_data=False):
        super(XingChallenge2017Reader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("XingChallenge2017Reader: Loading original data")


        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "xing_challenge_data_2017.zip"


        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            interactions_path = dataFile.extract("data/interactions_14.csv", path=decompressed_zip_file_folder + "decompressed/")

            ICM_path = dataFile.extract("data/items.csv", path=decompressed_zip_file_folder + "decompressed/")
            #UCM_path = dataFile.extract("data/users.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("XingChallenge2017Reader: Unable to find or extract data zip file.")
            print("XingChallenge2017Reader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            print("XingChallenge2017Reader: Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        print("XingChallenge2017Reader: Loading item content")
        ICM_all, feature_mapper, item_mapper = self._load_ICM(ICM_path, if_new_item="add")

        print("XingChallenge2017Reader: Loading Interactions")
        URM_all, _, user_mapper = self._load_interactions(interactions_path, item_mapper, if_new_user="add", if_new_item="ignore")

        print("XingChallenge2017Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        print("XingChallenge2017Reader: loading complete")

        return Dataset(self.get_dataset_name(), URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_all": ICM_all},
                       ICM_mappers_dict={"ICM_all": (item_mapper.copy(), feature_mapper.copy())})


    def _load_interactions(self, impressions_path, item_mapper, if_new_user ="add", if_new_item ="ignore"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_mapper, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = open(impressions_path, "r")
        numCells = 0

        # Remove header
        fileHandle.readline()

        for line in fileHandle:

            if numCells % 1000000 == 0 and numCells!=0:
                print("Processed {} cells".format(numCells))

            line = line.split("\t")

            """
            
            Interactions that the user performed on the job posting items. Fields:

            user_id ID          of the user who performed the interaction (points to users.id)
            item_id ID          of the item on which the interaction was performed (points to items.id)
            created_at          a unix time stamp timestamp representing the time when the interaction got created
            interaction_type    the type of interaction that was performed on the item:
                0 = XING showed this item to a user (= impression)
                1 = the user clicked on the item
                2 = the user bookmarked the item on XING
                3 = the user clicked on the reply button or application form button that is shown on some job postings
                4 = the user deleted a recommendation from his/her list of recommendation (clicking on "x") which has the effect that the recommendation will no longer been shown to the user and that a new recommendation item will be loaded and displayed to the user
                5 = a recruiter from the items company showed interest into the user. (e.g. clicked on the profile)
                        
                        
            """

            user_id = line[0]
            item_id = line[1]

            # transform negative interactions into a negative number
            interaction_type = int(line[2])
            if interaction_type == 4:
                interaction_type = -1
            elif interaction_type == 4:
                interaction_type = 0

            created_at = line[3]

            URM_builder.add_data_lists([user_id], [item_id], [interaction_type])

            numCells += 1

        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


    def _load_ICM(self, ICM_path, if_new_item ="ignore"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=None, on_new_row=if_new_item)

        fileHandle = open(ICM_path, "r")
        numCells = 0

        # Remove header
        # item_id	title	career_level	discipline_id	industry_id	country	is_payed	region	latitude	longitude	employment	tags	created_at
        fileHandle.readline()

        for line in fileHandle:

            if numCells % 1000000 == 0 and numCells!=0:
                print("Processed {} cells".format(numCells))

            line = line.split("\t")

            """
            id anonymized ID        of the item (referenced as item_id in the other datasets above)
            industry_id             anonymized IDs represent industries such as "Internet", "Automotive", "Finance", etc.
            discipline_id           anonymized IDs represent disciplines such as "Consulting", "HR", etc.
            is_paid (or is_payed)   indicates that the posting is a paid for by a compnay
            career_level            career level ID (e.g. beginner, experienced, manager)
                0 = unknown
                1 = Student/Intern
                2 = Entry Level (Beginner)
                3 = Professional/Experienced
                4 = Manager (Manager/Supervisor)
                5 = Executive (VP, SVP, etc.)
                6 = Senior Executive (CEO, CFO, President)
            country                 code of the country in which the job is offered
            latitude                latitude information (rounded to ca. 10km)
            longitude               longitude information (rounded to ca. 10km)
            region                  is specified for some users who have as country `de`. Meaning of the regions: see below.
            employment              the type of emploment
                0 = unknown
                1 = full-time
                2 = part-time
                3 = freelancer
                4 = intern
                5 = voluntary
            created_at              a unix time stamp timestamp representing the time when the interaction got created
            title                   concepts that have been extracted from the job title of the job posting (numeric IDs)
            tags                    concepts that have been extracted from the tags, skills or company name
            """

            item_id = line[0]

            title_id_list = line[1]
            title_id_list = title_id_list.split(",")

            career_level = line[2]

            discipline_id_list = line[3]
            discipline_id_list = discipline_id_list.split(",")

            industry_id_list = line[4]
            industry_id_list = industry_id_list.split(",")

            country = line[5]
            is_paid = line[6]

            region = line[7]

            latitude = line[8]
            longitude = line[9]

            employment = line[9]

            tags_list = line[10]
            tags_list = tags_list.split(",")

            created_at = line[11]

            item_token_list = [*title_id_list,  career_level,  *industry_id_list, *discipline_id_list, country, is_paid, region, employment, *tags_list]

            ICM_builder.add_single_row(item_id, item_token_list, data=1.0)

            numCells += 1

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

