#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

"""


import zipfile, os
import gzip

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder
from RecSysFramework.Utils import tagFilter, tagFilterAndStemming, IncrementalSparseMatrix_FilterIDs

from RecSysFramework.DataManager import Dataset


def _overlapping_feature(value, size=1, step=1):
    features = [int(value)]
    for i in range(1, size + 1):
        features.append(int(value) + i * step)
        features.append(int(value) - i * step)
    return features



class SteamReader(DataReader):

    DATASET_URL_REVIEWS = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
    DATASET_URL_METADATA = "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz"
    AVAILABLE_ICM = ["ICM_all"]
    DATASET_SUBFOLDER = "Steam/"


    def __init__(self, reload_from_original_data=False):
        super(SteamReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("Steam: Loading original data")

        folder_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not os.path.exists(folder_path + "steam_reviews.json.gz"):
            downloadFromURL(self.DATASET_URL_REVIEWS, folder_path, "steam_reviews.json.gz")

        if not os.path.exists(folder_path + "steam_games.json.gz"):
            downloadFromURL(self.DATASET_URL_METADATA, folder_path, "steam_games.json.gz")

        URM_path = folder_path + "steam_reviews.json.gz"
        ICM_path = folder_path + "steam_games.json.gz"

        URM_all, item_mapper, user_mapper = self._loadURM(URM_path, None, separator="\t", if_new_item="add", header=False)
        ICM_all, feature_mapper, item_mapper = self._loadICM(ICM_path, item_mapper, header=False)

        print("Steam: loading complete")

        return Dataset(
            self.get_dataset_name(),
            URM_dict={"URM_all": URM_all},
            URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
            ICM_dict={"ICM_all": ICM_all},
            ICM_mappers_dict={"ICM_all": (item_mapper.copy(), feature_mapper.copy())},
        )


    def _loadICM(self, tags_path, item_mapper, header=True, if_new_item="ignore"):

        ICM_builders = [IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                         preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)
                        for _ in range(13)]

        fileHandle = gzip.open(tags_path, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            row = eval(line)

            if "id" not in row.keys():
                continue


            movie_id = row['id']
            features = []

            if 'app_name' in row.keys():
                features.append(tagFilter(row['app_name'].lower()))
            else:
                features.append(None)

            if 'title' in row.keys():
                features.append(tagFilter(row['title'].lower()))
            else:
                features.append(None)

            if 'tags' in row.keys():
                features.append([x.lower().strip() for x in row['tags']])
            else:
                features.append(None)

            if 'specs' in row.keys():
                features.append([x.lower().strip() for x in row['specs']])
            else:
                features.append(None)

            if 'publisher' in row.keys():
                features.append([row['publisher'].lower().strip()])
            else:
                features.append(None)

            if 'developer' in row.keys():
                features.append([row['developer'].lower().strip()])
            else:
                features.append(None)

            if 'sentiment' in row.keys():
                features.append(tagFilterAndStemming(row['sentiment'].lower().strip()))
            else:
                features.append(None)

            if 'genres' in row.keys():
                features.append([x.lower().strip() for x in row['genres']])
            else:
                features.append(None)

            if 'early_access' in row.keys():
                features.append([str(row['early_access']).lower().strip()])
            else:
                features.append(None)

            if 'price' in row.keys():
                if 'free' in str(row['price']).lower():
                    row['price'] = 0
                try:
                    price = int(row['price']) % 5
                except Exception:
                    price = None
                features.append([price])
            else:
                features.append(None)

            if 'discount_price' in row.keys() and 'price' in row.keys() and price is not None:
                features.append([round(float(row['discount_price'])) != round(float(row['price']))])
            else:
                features.append(None)

            if 'release_date' in row.keys():
                try:
                    year = _overlapping_feature(int(row['release_date'].split("-")[0]), size=2, step=1)
                except Exception:
                    year = None
                features.append(year)
            else:
                features.append(None)

            if 'metascore' in row.keys():
                try:
                    metascore = int(row['metascore']) % 5
                except Exception:
                    metascore = None
                features.append([metascore])
            else:
                features.append(None)

            assert len(features) == len(ICM_builders), "Steam: Unexpected number of features ({}/{})" \
                                                       .format(len(features), len(ICM_builders))

            for i, ICM_builder in enumerate(ICM_builders):
                if features[i] is not None:
                    ICM_builder.add_single_row(movie_id, features[i], data=1.0)

        fileHandle.close()

        ICM_all = ICM_builders[0].get_SparseMatrix()
        feature_mapper = ICM_builders[0].get_column_token_to_id_mapper()

        print("Steam: Merging features in a single ICM")
        for ICM_builder in ICM_builders[1:]:
            ICM_all, feature_mapper = self._merge_ICM(ICM_all, ICM_builder.get_SparseMatrix(),
                            feature_mapper, ICM_builder.get_column_token_to_id_mapper())

        return ICM_all, feature_mapper, item_mapper


    def _loadURM(self, filePath, item_mapper, header=False, separator="::", if_new_user="add", if_new_item="ignore"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_mapper, on_new_col=if_new_item,
                                                        preinitialized_row_mapper=None, on_new_row=if_new_user)

        fileHandle = gzip.open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            row = eval(line)

            if set(['username', 'product_id', 'hours']).issubset(row.keys()):
                user_id = row['username']
                item_id = row['product_id']

                try:
                    value = float(row['hours'])
                    if value != 0.0:
                        URM_builder.add_data_lists([user_id], [item_id], [value])

                except:
                    pass

        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()
