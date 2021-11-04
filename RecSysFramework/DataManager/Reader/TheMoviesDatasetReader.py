#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import zipfile
import ast, csv, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager import Dataset
from RecSysFramework.DataManager.Utils import removeFeatures
from RecSysFramework.DataManager.DatasetPostprocessing import CombinedKCore

from RecSysFramework.Utils import reshapeSparse, IncrementalSparseMatrix_FilterIDs, invert_dictionary



class TheMoviesDatasetReader(DataReader):

    #DATASET_URL = "https://www.kaggle.com/rounakbanik/the-movies-dataset"
    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EQAIIMiVSTpIjZYmDqMyvukB8dur9LJ5cRT83CzXpLZ0TQ?e=lRNtWF"
    DATASET_SUBFOLDER = "TheMoviesDataset/"
    AVAILABLE_ICM = ["ICM_all", "ICM_credits", "ICM_metadata"]
    MANDATORY_POSTPROCESSINGS = [CombinedKCore(user_k_core=1, item_urm_k_core=1, item_icm_k_core=1, feature_k_core=1)]


    def __init__(self, reload_from_original_data=False):
        super(TheMoviesDatasetReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        print("TheMoviesDatasetReader: Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        zipFile_name = "the-movies-dataset.zip"

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + zipFile_name)

            credits_path = dataFile.extract("credits.csv", path=decompressed_zip_file_folder + "decompressed/")
            metadata_path = dataFile.extract("movies_metadata.csv", path=decompressed_zip_file_folder + "decompressed/")
            movielens_tmdb_id_map_path = dataFile.extract("links.csv", path=decompressed_zip_file_folder + "decompressed/")

            URM_path = dataFile.extract("ratings.csv", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("TheMoviesDatasetReader: Unable to find or extract data zip file.")
            print("TheMoviesDatasetReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            print("TheMoviesDatasetReader: Data zip file not found or damaged. You may download the data from: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        self.item_original_ID_to_title = {}
        self.item_index_to_title = {}

        print("TheMoviesDatasetReader: Loading ICM_credits")
        ICM_credits, ICM_credits_mapper, item_mapper = self._loadICM_credits(credits_path, header=True, if_new_item="add")

        print("TheMoviesDatasetReader: Loading ICM_metadata")
        ICM_metadata, ICM_metadata_mapper, item_mapper = self._loadICM_metadata(metadata_path, item_mapper,
                                                                                header=True, if_new_item="add")

        ICM_credits, _, ICM_credits_mapper = removeFeatures(ICM_credits, minOccurrence=5, maxPercOccurrence=0.30,
                                                            reconcile_mapper=ICM_credits_mapper)

        ICM_metadata, _, ICM_metadata_mapper = removeFeatures(ICM_metadata, minOccurrence=5, maxPercOccurrence=0.30,
                                                              reconcile_mapper=ICM_metadata_mapper)

        n_items = ICM_metadata.shape[0]

        ICM_credits = reshapeSparse(ICM_credits, (n_items, ICM_credits.shape[1]))

        # IMPORTANT: ICM uses TMDB indices, URM uses movielens indices
        # Load index mapper
        movielens_id_to_tmdb, tmdb_to_movielens_id = self._load_item_id_mapping(movielens_tmdb_id_map_path, header=True)

        # Modify saved mapper to accept movielens id instead of tmdb
        item_mapper = self._replace_tmdb_id_with_movielens(tmdb_to_movielens_id, item_mapper)

        print("TheMoviesDatasetReader: Loading URM")
        URM_all, _, user_mapper = self._load_URM(URM_path, item_mapper, header=True, separator=",",
                                                 if_new_user="add", if_new_item="ignore")

        # Reconcile URM and ICM
        # Keep only items having ICM entries, remove all the others
        n_items = ICM_credits.shape[0]
        URM_all = URM_all[:, 0:n_items]

        # URM is already clean

        ICM_all, ICM_all_mapper = self._merge_ICM(ICM_credits, ICM_metadata, ICM_credits_mapper, ICM_metadata_mapper)

        print("TheMoviesDatasetReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        print("TheMoviesDatasetReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_credits": ICM_credits, "ICM_metadata": ICM_metadata, "ICM_all": ICM_all},
                       ICM_mappers_dict={"ICM_credits": (item_mapper.copy(), ICM_credits_mapper.copy()),
                                         "ICM_metadata": (item_mapper.copy(), ICM_metadata_mapper.copy()),
                                         "ICM_all": (item_mapper.copy(), ICM_all_mapper.copy())})


    def _load_item_id_mapping(self, movielens_tmdb_id_map_path, header=True):

        movielens_id_to_tmdb = {}
        tmdb_to_movielens_id = {}

        movielens_tmdb_id_map_file = open(movielens_tmdb_id_map_path, 'r', encoding="utf8")

        if header:
            movielens_tmdb_id_map_file.readline()

        for newMapping in movielens_tmdb_id_map_file:

            newMapping = newMapping.split(",")

            movielens_id = newMapping[0]
            tmdb_id = newMapping[2].replace("\n", "")

            movielens_id_to_tmdb[movielens_id] = tmdb_id
            tmdb_to_movielens_id[tmdb_id] = movielens_id

        return movielens_id_to_tmdb, tmdb_to_movielens_id


    def _replace_tmdb_id_with_movielens(self, tmdb_to_movielens_id, item_mapper):
        """
        Replace 'the original id' in such a way that it points to the same index
        :param tmdb_to_movielens_id:
        :return:
        """

        item_original_ID_to_index_movielens = {}
        item_index_to_original_ID_movielens = {}
        item_original_ID_to_title_movielens = {}

        self.item_index_to_original_ID = invert_dictionary(item_mapper)

        for item_index in self.item_index_to_original_ID.keys():

            tmdb_id = self.item_index_to_original_ID[item_index]

            if tmdb_id in self.item_original_ID_to_title:
                movie_title = self.item_original_ID_to_title[tmdb_id]
            else:
                movie_title = ""

            movielens_id = tmdb_to_movielens_id[tmdb_id]
            item_index_to_original_ID_movielens[item_index] = movielens_id
            item_original_ID_to_index_movielens[movielens_id] = item_index
            item_original_ID_to_title_movielens[movielens_id] = movie_title

        # Replace the TMDB based mapper
        self.item_index_to_original_ID = item_index_to_original_ID_movielens
        self.item_original_ID_to_title = item_original_ID_to_title_movielens

        return item_original_ID_to_index_movielens


    def _loadICM_credits(self, credits_path, header=True, if_new_item = "add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)

        numCells = 0
        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')

        for newCredits in parser_credits:

            # newCredits is a tuple of two strings, both are lists of dictionaries
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # NOTE: sometimes a dict value is ""Savannah 'Vannah' Jackson"", if the previous eval removes the commas "" "" then the parsing of the string will fail
            cast_list = []
            credits_list = []

            try:
                cast_list = ast.literal_eval(newCredits[0])
                credits_list = ast.literal_eval(newCredits[1])
            except Exception as e:
                print("TheMoviesDatasetReader: Exception while parsing: '{}', skipping".format(str(e)))

            movie_id = newCredits[2]
            cast_list.extend(credits_list)
            cast_list_name = [cast_member["name"] for cast_member in cast_list]
            ICM_builder.add_single_row(movie_id, cast_list_name, data=1.0)

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


    def _loadICM_metadata(self, metadata_path, item_mapper, header=True, if_new_item="add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)

        numCells = 0
        metadata_file = open(metadata_path, 'r', encoding="utf8")

        if header:
            metadata_file.readline()

        parser_metadata = csv.reader(metadata_file, delimiter=',', quotechar='"')

        for newMetadata in parser_metadata:

            numCells += 1
            if numCells % 100000 == 0:
                print("Processed {} cells".format(numCells))

            token_list = []

            if len(newMetadata) < 22:
                #Sono 6, ragionevole
                print("TheMoviesDatasetReader: Line too short, possible unwanted new line character, skipping")
                continue

            movie_id = newMetadata[5]


            if newMetadata[0] == "True":
                token_list.append("ADULTS_YES")
            else:
                token_list.append("ADULTS_NO")

            if newMetadata[1]:
                collection = ast.literal_eval(newMetadata[1])
                token_list.append("collection_" + str(collection["id"]))

            #budget = int(rating[2])

            if newMetadata[3]:
                genres = ast.literal_eval(newMetadata[3])

                for genre in genres:
                    token_list.append("genre_" + str(genre["id"]))

            orig_lang = newMetadata[7]
            title = newMetadata[8]

            if movie_id not in self.item_original_ID_to_title:
                self.item_original_ID_to_title[movie_id] = title

            if orig_lang:
                token_list.append("original_language_"+orig_lang)

            if newMetadata[12]:
                prod_companies = ast.literal_eval(newMetadata[12])
                for prod_company in prod_companies:
                    token_list.append("production_company_" + str(prod_company['id']))

            if newMetadata[13]:
                prod_countries = ast.literal_eval(newMetadata[13])
                for prod_country in prod_countries:
                    token_list.append("production_country_" + prod_country['iso_3166_1'])

            try:
                release_date = int(newMetadata[14].split("-")[0])
                token_list.append("release_date_" + str(release_date))
            except Exception:
                pass

            if newMetadata[17]:
                spoken_langs = ast.literal_eval(newMetadata[17])
                for spoken_lang in spoken_langs:
                    token_list.append("spoken_lang_" + spoken_lang['iso_639_1'])

            if newMetadata[18]:
                status = newMetadata[18]
                if status:
                    token_list.append("status_" + status)

            if newMetadata[21] == "True":
                token_list.append("VIDEO_YES")
            else:
                token_list.append("VIDEO_NO")

            ICM_builder.add_single_row(movie_id, token_list, data=True)

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


    def _load_URM(self, filePath, item_mapper, header=False, separator="::", if_new_user="add", if_new_item="ignore"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_mapper, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)
                line[-1] = line[-1].replace("\n", "")

                try:
                    user_id = line[0]
                    item_id = line[1]
                    try:
                        value = float(line[2])
                        if value != 0.0:
                            URM_builder.add_data_lists([user_id], [item_id], [value])

                    except ValueError:
                        print("load_CSV_into_SparseBuilder: Cannot parse as float value '{}'".format(line[2]))

                except IndexError:
                    print("load_CSV_into_SparseBuilder: Index out of bound in line '{}'".format(line))

        fileHandle.close()

        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()

