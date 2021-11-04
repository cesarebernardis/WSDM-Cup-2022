#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""


import ast, gzip, os, itertools, json

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder, removeFeatures

from RecSysFramework.Utils import IncrementalSparseMatrix_FilterIDs
from RecSysFramework.Utils import tagFilterAndStemming

from RecSysFramework.DataManager import Dataset


from collections import Iterable

def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def parse_json(file_path):

    if file_path.endswith(".gz"):
        g = gzip.open(file_path, 'r')
    else:
        g = open(file_path, 'r')

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))


def parse_json_new(file_path):

    if file_path.endswith(".gz"):
        g = gzip.open(file_path, 'r')
    else:
        g = open(file_path, 'r')
    
    for l in g:
        try:
            yield json.loads(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))


class AmazonReviewDataReader(DataReader):

    DATASET_SUBFOLDER = "AmazonReviewData/"


    def __init__(self, reload_from_original_data=False):
        super(AmazonReviewDataReader, self).__init__(reload_from_original_data)


    def _get_ICM_metadata_path(self, data_folder, compressed_file_name, decompressed_file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """

        try:

            open(data_folder + decompressed_file_name, "r")

        except FileNotFoundError:

            print("AmazonReviewDataReader: Decompressing metadata file...")

            try:

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()

            except (FileNotFoundError, Exception):

                print("AmazonReviewDataReader: Unable to find or decompress compressed file. Downloading...")

                downloadFromURL(file_url, data_folder, compressed_file_name)

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()

        return data_folder + decompressed_file_name


    def _get_URM_review_path(self, data_folder, file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """

        try:

            open(data_folder + file_name, "r")

        except FileNotFoundError:

            print("AmazonReviewDataReader: Unable to find or open review file. Downloading...")

            downloadFromURL(file_url, data_folder, file_name)

        return data_folder + file_name


    def _load_from_original_file_all_amazon_datasets(self, URM_path, metadata_path=None, reviews_path=None, urm_from_reviews=False):

        print("AmazonReviewDataReader: Loading original data")

        print("AmazonReviewDataReader: loading URM")
        if urm_from_reviews:
            URM_all, item_original_ID_to_index, user_original_ID_to_index = self._loadURMfromReviews(URM_path, None, if_new_item="add")
        else:
            URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=",",
                                                                                                        header=False, remove_duplicates=True)

        print(URM_all.shape, URM_all.data.size)
        urm = {"URM_all": URM_all}
        urm_mappers = {"URM_all": (user_original_ID_to_index, item_original_ID_to_index)}

        icm = {}
        icm_mappers = {}
        if metadata_path is not None:
            print("AmazonReviewDataReader: loading metadata")
            ICM_metadata, feature_mapper, item_mapper = self._loadMetadata(metadata_path, item_original_ID_to_index, if_new_item="ignore")
            ICM_metadata, _, feature_mapper = removeFeatures(ICM_metadata, minOccurrence=5, maxPercOccurrence=0.30,
                                                             reconcile_mapper=feature_mapper)
            icm["ICM_metadata"] = ICM_metadata
            icm_mappers["ICM_metadata"] = (item_mapper.copy(), feature_mapper.copy())

        if reviews_path is not None:
            print("AmazonReviewDataReader: loading reviews")
            ICM_reviews, feature_mapper, item_mapper = self._loadReviews(reviews_path, item_original_ID_to_index, if_new_item="ignore")
            ICM_reviews, _, feature_mapper = removeFeatures(ICM_reviews, minOccurrence=5, maxPercOccurrence=0.30,
                                                            reconcile_mapper=feature_mapper)
            icm["ICM_reviews"] = ICM_reviews
            icm_mappers["ICM_reviews"] = (item_mapper.copy(), feature_mapper.copy())

        if len(icm) > 0:
            ICM_names = list(icm.keys())
            ICM_all, ICM_all_mapper = icm[ICM_names[0]], icm_mappers[ICM_names[0]]
            for key in ICM_names[1:]:
                ICM_all, ICM_all_mapper = self._merge_ICM(ICM_all, icm[key], ICM_all_mapper, icm_mappers[key])
            icm["ICM_all"] = ICM_all
            icm_mappers["ICM_all"] = ICM_all_mapper

        # Clean temp files
        print("AmazonReviewDataReader: cleaning temporary files")

        if metadata_path is not None:
            os.remove(metadata_path)

        if reviews_path is not None:
            os.remove(reviews_path)

        print("AmazonReviewDataReader: loading complete")

        return Dataset(self.get_dataset_name(),
                       URM_dict=urm, URM_mappers_dict=urm_mappers,
                       ICM_dict=icm, ICM_mappers_dict=icm_mappers)


    def _loadMetadata(self, file_path, item_mapper, if_new_item="ignore"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)

        parser_metadata = parse_json(file_path)

        for numMetadataParsed, newMetadata in enumerate(parser_metadata):

            if numMetadataParsed % 20000 == 0:
                print("Processed {}".format(numMetadataParsed))

            item_ID = newMetadata["asin"]

            # The file might contain other elements, restrict to
            # Those in the URM

            tokenList = []

            if "title" in newMetadata:
                item_name = newMetadata["title"]
                tokenList.append(item_name)

            # Sometimes brand is not present
            if "brand" in newMetadata:
                item_brand = newMetadata["brand"]
                tokenList.append(item_brand)

            # Categories are a list of lists. Unclear whether only the first element contains data or not
            if "categories" in newMetadata:
                item_categories = flatten(newMetadata["categories"])
                tokenList.extend(item_categories)

            # List of strings
            if "feature" in newMetadata:
                tokenList.extend(newMetadata["feature"])

            if "description" in newMetadata:
                item_description = newMetadata["description"]
                if isinstance(item_description, str):
                    tokenList.append(item_description)
                else:
                    tokenList.extend(item_description)

            #print(tokenList)
            tokenList = ' '.join(tokenList).lower().strip()

            # Remove non alphabetical character and split on spaces
            tokenList = tagFilterAndStemming(tokenList)

            # Remove duplicates
            tokenList = list(set(tokenList))

            ICM_builder.add_single_row(item_ID, tokenList, data=1.0)

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


    def _loadURMfromReviews(self, file_path, item_mapper, if_new_item="add"):

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)
 
        parser_reviews = parse_json_new(file_path)

        for numReviewParsed, newReview in enumerate(parser_reviews):

            if numReviewParsed % 100000 == 0:
                print("Processed {} reviews".format(numReviewParsed))

            user_ID = newReview["reviewerID"]
            item_ID = newReview["asin"]
            rating = float(newReview["overall"])
            #print(user_ID, item_ID, rating)

            URM_builder.add_single_row(user_ID, [item_ID], rating)

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


    def _loadReviews(self, file_path, item_mapper, if_new_item="add"):

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=None, on_new_col="add",
                                                        preinitialized_row_mapper=item_mapper, on_new_row=if_new_item)

        parser_reviews = parse_json(file_path)

        numReviewParsed = 0

        for newReview in parser_reviews:

            numReviewParsed+=1
            if (numReviewParsed % 20000 == 0):
                print("Processed {} reviews".format(numReviewParsed))

            user_ID = newReview["reviewerID"]
            item_ID = newReview["asin"]

            reviewText = newReview["reviewText"]
            reviewSummary = newReview["summary"]

            tagList = ' '.join([reviewText, reviewSummary])

            # Remove non alphabetical character and split on spaces
            tagList = tagFilterAndStemming(tagList)

            ICM_builder.add_single_row(item_ID, tagList, data=1.0)

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

