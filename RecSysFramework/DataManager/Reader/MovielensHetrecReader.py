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


class MovielensHetrecReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
    DATASET_SUBFOLDER = "MovielensHetrec/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_years", "ICM_directors", "ICM_actors", "ICM_countries",
                     "ICM_locations"]


    def __init__(self, reload_from_original_data=False):
        super(MovielensHetrecReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("MovielensHetrecReader: Loading original data")

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("MovielensHetrecReader: Unable to fild data zip file. Downloading...")
            downloadFromURL(self.DATASET_URL, zipFile_path, "hetrec2011-movielens-2k-v2.zip")
            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")

        movies_path = dataFile.extract("movies.dat", path=zipFile_path + "decompressed/")
        genres_path = dataFile.extract("movie_genres.dat", path=zipFile_path + "decompressed/")
        directors_path = dataFile.extract("movie_directors.dat", path=zipFile_path + "decompressed/")
        actors_path = dataFile.extract("movie_actors.dat", path=zipFile_path + "decompressed/")
        countries_path = dataFile.extract("movie_countries.dat", path=zipFile_path + "decompressed/")
        locations_path = dataFile.extract("movie_locations.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("user_ratedmovies.dat", path=zipFile_path + "decompressed/")

        print("MovielensHetrecReader: loading years")
        ICM_years, years_mapper, item_mapper = self._load_tsv(movies_path, None, feature_columns=[5], header=True,
                                                              if_new_item="add")

        print("MovielensHetrecReader: loading genres")
        ICM_genres, genres_mapper, _ = self._load_tsv(genres_path, item_mapper, header=True, if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_genres, ICM_years, genres_mapper, years_mapper)

        print("MovielensHetrecReader: loading directors")
        ICM_directors, directors_mapper, _ = self._load_tsv(directors_path, item_mapper, header=True,
                                                            if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_directors, feature_mapper, directors_mapper)

        print("MovielensHetrecReader: loading actors")
        ICM_actors, actors_mapper, _ = self._load_tsv(actors_path, item_mapper, header=True, if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_actors, feature_mapper, actors_mapper)

        print("MovielensHetrecReader: loading countries")
        ICM_countries, countries_mapper, _ = self._load_tsv(countries_path, item_mapper, header=True,
                                                            if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_countries, feature_mapper, countries_mapper)

        print("MovielensHetrecReader: loading locations")
        ICM_locations, locations_mapper, _ = self._load_tsv(locations_path, item_mapper, feature_columns=[1, 2, 3],
                                                            header=True, if_new_item="ignore")
        ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_locations, feature_mapper, locations_mapper)

        print("MovielensHetrecReader: loading URM")
        URM_all, _, user_mapper = self._loadURM(URM_path, item_mapper, separator="\t", header=True,
                                                if_new_user="add", if_new_item="ignore")

        print("MovielensHetrecReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("MovielensHetrecReader: saving URM and ICM")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={
                           "ICM_genres": ICM_genres, "ICM_years": ICM_years, "ICM_all": ICM_all,
                           "ICM_directors": ICM_directors, "ICM_actors": ICM_actors, "ICM_countries": ICM_countries,
                           "ICM_locations": ICM_locations,
                       },
                       ICM_mappers_dict={"ICM_genres": (item_mapper.copy(), genres_mapper.copy()),
                                         "ICM_years": (item_mapper.copy(), years_mapper.copy()),
                                         "ICM_directors": (item_mapper.copy(), directors_mapper.copy()),
                                         "ICM_actors": (item_mapper.copy(), actors_mapper.copy()),
                                         "ICM_countries": (item_mapper.copy(), countries_mapper.copy()),
                                         "ICM_locations": (item_mapper.copy(), locations_mapper.copy()),
                                         "ICM_all": (item_mapper.copy(), feature_mapper.copy())})


    def _load_tsv(self, tags_path, item_mapper, feature_columns=None, filter_and_stem=False,
                  header=True, separator='\t', if_new_item="ignore"):

        if feature_columns is None:
            feature_columns = [1]

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
                movie_id = line[0]

                for col in feature_columns:
                    tagList = line[col].lower()

                    # Remove non alphabetical character and split on spaces
                    if filter_and_stem:
                        tagList = tagFilterAndStemming(tagList)
                    else:
                        tagList = [tagList]

                    # Rows movie ID
                    # Cols features
                    ICM_builder.add_single_row(movie_id, tagList, data=1.0)

        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


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
                print("Processed {} cells".format(numCells))

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
