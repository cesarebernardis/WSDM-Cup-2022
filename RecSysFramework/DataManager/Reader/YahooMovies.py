#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

"""


import zipfile, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder
from RecSysFramework.Utils import tagFilter, tagFilterAndStemming, IncrementalSparseMatrix_FilterIDs

from RecSysFramework.DataManager import Dataset


class YahooMoviesReader(DataReader):

    AVAILABLE_ICM = ["ICM_all"]
    DATASET_SUBFOLDER = "YahooMovies/"


    def __init__(self, reload_from_original_data=False):
        super(YahooMoviesReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("YahooMovies: Loading original data")

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "yahoo-movies-dataset.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("YahooMovies: Unable to find data zip file.")
            print("YahooMovies: Automatic download not available, " \
                  "please ensure the ZIP data file is in folder {}.".format(zipFile_path))

            # If directory does not exist, create
            if not os.path.exists(zipFile_path):
                os.makedirs(zipFile_path)

            raise FileNotFoundError("Automatic download not available.")


        URM_path = dataFile.extract("dataset/ydata-ymovies-user-movie-ratings.txt", path=zipFile_path + "decompressed/")
        ICM_path = dataFile.extract("dataset/ydata-ymovies-movie-content-descr.txt", path=zipFile_path + "decompressed/")

        URM_all, item_mapper, user_mapper = self._loadURM(URM_path, None, separator="\t", if_new_item="add", header=False)
        ICM_all, feature_mapper, item_mapper = self._loadICM(ICM_path, item_mapper, header=False)

        print("YahooMovies: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("YahooMovies: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_all": ICM_all},
                       ICM_mappers_dict={"ICM_all": (item_mapper.copy(), feature_mapper.copy())},
                       )


    def _loadICM(self, tags_path, item_mapper, feature_columns=None, filter_and_stem=False, null_character="\\N",
                  header=True, separator='\t', if_new_item="ignore"):

        if feature_columns is None:
            feature_columns = [1]

        ICM_builders = [IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                          preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)
                        for _ in range(10)]

        fileHandle = open(tags_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        """
            ORIGINAL DESCRIPTION IS WRONG!
        
            0 Yahoo! movie id
            1 title
            2 synopsis   
            3 running time
            4 MPAA rating  
            5 reasons for the MPAA rating
            6 complete release date
            7 release date (yyyymmdd)
            8 distributor
    
            9 URL of poster. To construct a valid URL, append field to
              "http://us.movies1.yimg.com/movies.yahoo.com/images/hv/"
    
            10 list of genres
            11 list of directors
            12 list of director ids
            13 list of types of crew
            14 list of crew members
            15 list of crew ids
    
            16 list of actors (name [character]). Character name information
               may be incomplete or incorrect.
    
            17 list of actor ids. To construct a URL to the Yahoo! Movies Main
               Page for an actor, append the actor id to
               "http://movies.yahoo.com/movie/contributor/",
               e.g. http://movies.yahoo.com/movie/contributor/1800019596
    
            18 average critic rating
            19 the number of critic ratings
            20 the number of awards won
            21 the number of awards nominated
            22 list of awards won
            23 list of awards nominated
    
            24 rating from The Movie Mom. More information about The Movie Mom
            is available at http://movies.yahoo.com/mv/moviemom/about.html
    
            25 review from The Movie Mom. More information about The Movie Mom
            is available at http://movies.yahoo.com/mv/moviemom/about.html
    
            26 list of review summaries by critics and users
            27 list of anonymized review owners
            28 list of captions from trailers/clips
    
            29 URL of Greg's Preview. To construct a valid URL, append field
               to "http://movies.yahoo.com/movie/preview/". Greg's Previews of
               Upcoming Movies are compiled by Greg Dean Schmitz (from
               UpcomingMovies.com). Greg's Previews are available at
               http://movies.yahoo.com/mv/upcoming/ . More information about
               Greg's Previews is available at
               http://movies.yahoo.com/feature/aboutgreg.html
    
            30 URL of DVD review. To construct a valid URL, append field to
               "http://movies.yahoo.com/mv/dvd/reviews/"
    
            31 global non-personalized popularity (GNPP)
    
               GNPP was generated by Yahoo! Research as follows:
               GNPP = 1/k *
               (avg(i)+log_2(n(i))+log_10(#awards_won*10+#award_nomination*5))
               where avg(i) is field 31, n(i) is field 32, and k is
               normalization factor such that the maximum GNPP value is 13.
    
            32 average rating of this item among users in the training data
            33 the number of users in the training data who rated this item
        """

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if len(line) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                movie_id = line[0]

                features = []

                # Title
                if line[1] != null_character:
                    features.append(tagFilter(line[1].lower()))
                else:
                    features.append(None)

                # Synopsis
                if line[2] != null_character:
                    features.append(tagFilterAndStemming(line[2].lower()))
                else:
                    features.append(None)

                # Release year
                if line[7] != null_character:
                    features.append([int(line[7][:4])])
                else:
                    features.append(None)

                # Distributor
                if line[8] != null_character:
                    features.append([line[8].lower().strip()])
                else:
                    features.append(None)

                # List of genres
                if line[10] != null_character:
                    features.append([s.lower().strip() for s in line[10].split("|") if len(s) > 0])
                else:
                    features.append(None)

                # List of director ids
                if line[12] != null_character:
                    features.append([int(s) for s in line[12].split("|") if len(s) > 0])
                else:
                    features.append(None)

                # List of crew ids
                if line[15] != null_character:
                    features.append([int(s) for s in line[15].split("|") if len(s) > 0])
                else:
                    features.append(None)

                # List of actor ids
                if line[17] != null_character:
                    features.append([int(s) for s in line[17].split("|") if len(s) > 0])
                else:
                    features.append(None)

                # List of awards won
                if line[22] != null_character:
                    features.append([s.lower().strip() for s in line[22].split("|") if len(s) > 0])
                else:
                    features.append(None)

                # List of awards nominated
                if line[23] != null_character:
                    features.append([s.lower().strip() for s in line[23].split("|") if len(s) > 0])
                else:
                    features.append(None)

                assert len(features) == len(ICM_builders), "YahooMovies: Unexpected number of features ({}/{})" \
                                                           .format(len(features), len(ICM_builders))

                for i, ICM_builder in enumerate(ICM_builders):
                    if features[i] is not None:
                        ICM_builder.add_single_row(movie_id, features[i], data=1.0)

        fileHandle.close()

        ICM_all = ICM_builders[0].get_SparseMatrix()
        feature_mapper = ICM_builders[0].get_column_token_to_id_mapper()

        print("YahooMovies: Merging features in a single ICM")
        for ICM_builder in ICM_builders[1:]:
            ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_builder.get_SparseMatrix(),
                            feature_mapper, ICM_builder.get_column_token_to_id_mapper())

        return ICM_all, feature_mapper, item_mapper


    def _loadURM(self, filePath, item_mapper, header=False, separator="::", if_new_user="add", if_new_item="ignore"):

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
                value = float(line[3])
                if value != 0.0:
                    URM_builder.add_data_lists([user_id], [item_id], [value])

            except:
                pass

        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()
