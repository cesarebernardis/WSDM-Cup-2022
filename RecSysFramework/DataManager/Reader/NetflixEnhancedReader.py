#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import zipfile, os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import removeFeatures
from RecSysFramework.DataManager.DatasetPostprocessing import URMKCore

from RecSysFramework.DataManager import Dataset


class NetflixEnhancedReader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EYHOBNp2nF1Gtvm2ELd9eLkBJfwKvU2O4Cp9HjO6HUJkhA?e=I2S1OC"
    DATASET_SUBFOLDER = "NetflixEnhanced/"
    AVAILABLE_ICM = ["ICM_all", "ICM_tags", "ICM_editorial"]
    MANDATORY_POSTPROCESSINGS = [URMKCore(user_k_core=1, item_k_core=1)]


    def __init__(self, reload_from_original_data=False):
        super(NetflixEnhancedReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        print("NetflixEnhancedReader: Loading original data")

        compressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(compressed_zip_file_folder + "NetflixEnhancedData.zip")

            URM_matfile_path = dataFile.extract("urm.mat", path=decompressed_zip_file_folder + "decompressed/")
            titles_matfile_path = dataFile.extract("titles.mat", path=decompressed_zip_file_folder + "decompressed/")
            ICM_matfile_path = dataFile.extract("icm.mat", path=decompressed_zip_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("NetflixPrizeReader: Unable to find or extract data zip file.")
            print("NetflixPrizeReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(compressed_zip_file_folder))
            print("NetflixPrizeReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_zip_file_folder):
                os.makedirs(compressed_zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")

        URM_matfile = sio.loadmat(URM_matfile_path)

        URM_all = URM_matfile["urm"]
        usercache_urm = URM_matfile["usercache_urm"]
        itemcache_urm = URM_matfile["itemcache_urm"]

        user_mapper = {}
        item_mapper = {}

        for item_id in range(URM_all.shape[1]):
            item_mapper[item_id] = item_id

        for user_id in range(URM_all.shape[0]):
            user_mapper[user_id] = user_id

        titles_matfile = sio.loadmat(titles_matfile_path)

        titles_list = titles_matfile["titles"]

        ICM_matfile = sio.loadmat(ICM_matfile_path)

        ICM_all = ICM_matfile["icm"]
        ICM_all = sps.csr_matrix(ICM_all.T)

        ICM_dictionary = ICM_matfile["dictionary"]
        itemcache_icm = ICM_matfile["itemcache_icm"]
        stemTypes = ICM_dictionary["stemTypes"][0][0]
        stems = ICM_dictionary["stems"][0][0]

        # Split ICM_tags and ICM_editorial
        is_tag_mask = np.zeros((len(stems)), dtype=np.bool)

        ICM_all_mapper, ICM_tags_mapper, ICM_editorial_mapper = {}, {}, {}

        for current_stem_index in range(len(stems)):
            current_stem_type = stemTypes[current_stem_index]
            current_stem_type_string = current_stem_type[0][0]

            token = stems[current_stem_index][0][0]

            if token in ICM_all_mapper:
                print("Duplicate token {} alredy existent in position {}".format(token, ICM_all_mapper[token]))

            else:
                ICM_all_mapper[token] = current_stem_index
                if "KeywordsArray" in current_stem_type_string:
                    is_tag_mask[current_stem_index] = True
                    ICM_tags_mapper[token] = len(ICM_tags_mapper)
                else:
                    ICM_editorial_mapper[token] = len(ICM_editorial_mapper)

        ICM_tags = ICM_all[:, is_tag_mask]

        is_editorial_mask = np.logical_not(is_tag_mask)
        ICM_editorial = ICM_all[:, is_editorial_mask]

        # Remove features taking into account the filtered ICM
        ICM_all, _, ICM_all_mapper = removeFeatures(ICM_all, minOccurrence=5, maxPercOccurrence=0.30,
                                                    reconcile_mapper=ICM_all_mapper)
        ICM_tags, _, ICM_tags_mapper = removeFeatures(ICM_tags, minOccurrence=5, maxPercOccurrence=0.30,
                                                      reconcile_mapper=ICM_tags_mapper)
        ICM_editorial, _, ICM_editorial_mapper = removeFeatures(ICM_editorial, minOccurrence=5, maxPercOccurrence=0.30,
                                                                reconcile_mapper=ICM_editorial_mapper)

        print("NetflixEnhancedReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_zip_file_folder + "decompressed", ignore_errors=True)

        print("NetflixEnhancedReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict={"URM_all": URM_all},
                       URM_mappers_dict={"URM_all": (user_mapper.copy(), item_mapper.copy())},
                       ICM_dict={"ICM_editorial": ICM_editorial, "ICM_tags": ICM_tags, "ICM_all": ICM_all},
                       ICM_mappers_dict={"ICM_editorial": (item_mapper.copy(), ICM_editorial_mapper.copy()),
                                         "ICM_tags": (item_mapper.copy(), ICM_tags_mapper.copy()),
                                         "ICM_all": (item_mapper.copy(), ICM_all_mapper.copy())})
