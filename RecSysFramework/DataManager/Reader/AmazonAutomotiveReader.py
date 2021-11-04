#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""



from RecSysFramework.DataManager.Reader.AmazonReviewDataReader import AmazonReviewDataReader



class AmazonAutomotiveReader(AmazonReviewDataReader):

    DATASET_URL_RATING = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Automotive.csv"
    DATASET_URL_METADATA = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Automotive.json.gz"

    DATASET_SUBFOLDER = "AmazonReviewData/AmazonAutomotive/"
    AVAILABLE_ICM = ["ICM_all", "ICM_metadata"]


    def __init__(self, reload_from_original_data=False):
        super(AmazonAutomotiveReader, self).__init__(reload_from_original_data)


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        # Load data from original

        print("AmazonAutomotiveReader: Loading original data")

        dataset_split_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        metadata_path = self._get_ICM_metadata_path(data_folder=dataset_split_folder,
                                                    compressed_file_name="meta_Automotive.json.gz",
                                                    decompressed_file_name="meta_Automotive.json",
                                                    file_url=self.DATASET_URL_METADATA)

        URM_path = self._get_URM_review_path(data_folder=dataset_split_folder,
                                             file_name="ratings_Automotive.csv",
                                             file_url=self.DATASET_URL_RATING)

        return self._load_from_original_file_all_amazon_datasets(URM_path, metadata_path=metadata_path, reviews_path=None)


