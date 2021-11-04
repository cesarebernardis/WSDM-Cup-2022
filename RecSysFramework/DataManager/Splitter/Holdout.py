#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix

from .DataSplitter import DataSplitter


class Holdout(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def __init__(self, train_perc=0.8, validation_perc=0.0, test_perc=0.2, forbid_new_split=False,
                 test_rating_threshold=0, force_new_split=False, allow_cold_users=False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        assert train_perc + validation_perc + test_perc == 1, "DataSplitterHoldout: percentages do not sum to 1"

        super(Holdout, self).__init__(forbid_new_split=forbid_new_split, force_new_split=force_new_split,
                                      allow_cold_users=allow_cold_users, with_validation=validation_perc > 0)
        self.test_rating_threshold = test_rating_threshold
        self.train_perc = train_perc
        self.validation_perc = validation_perc
        self.test_perc = test_perc


    def get_name(self):
        return "holdout_{:.2f}_{:.2f}_{:.2f}_testtreshold_{:.1f}{}" \
               .format(self.train_perc, self.validation_perc, self.test_perc, self.test_rating_threshold,
                       "" if self.allow_cold_users else "_no_cold_users")


    def split(self, dataset, random_seed=42):

        super(Holdout, self).split(dataset, random_seed=random_seed)

        URM = sps.csr_matrix(dataset.get_URM())

        n_users, n_items = dataset.n_users, dataset.n_items
        user_indices = []
        URM_train, URM_test, URM_validation = {}, {}, {}

        #Select apriori how to randomizely sort every user
        users_to_remove = []
        for user_id in range(n_users):
            assignment = np.random.choice(3, URM.indptr[user_id + 1] - URM.indptr[user_id], replace=True,
                                          p=[self.train_perc, self.validation_perc, self.test_perc])
            assignments = [assignment == i for i in range(3)]
            #if assignments[2].sum() <= 0:
                #No interactions in test
            #    users_to_remove.append(user_id)
            #if self.with_validation and assignments[1].sum() <= 0:
                #No interactions in validation
            #    users_to_remove.append(user_id)
            if not self.allow_cold_users and assignments[0].sum() <= 0:
                #No interactions in train
                users_to_remove.append(user_id)
            user_indices.append(assignments)

        for URM_name in dataset.get_URM_names():

            URM = dataset.get_URM(URM_name)
            URM = sps.csr_matrix(URM)

            URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                        auto_create_col_mapper=False, n_cols=n_items)

            URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                       auto_create_col_mapper=False, n_cols=n_items)

            if self.with_validation:
                URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                                 auto_create_col_mapper=False, n_cols=n_items)

            users_to_remove_index = 0
            for user_id in range(n_users):

                if users_to_remove_index < len(users_to_remove) and user_id == users_to_remove[users_to_remove_index]:
                    users_to_remove_index += 1
                    continue

                indices = user_indices[user_id]

                start_user_position = URM.indptr[user_id]
                end_user_position = URM.indptr[user_id + 1]

                user_interaction_items = URM.indices[start_user_position:end_user_position]
                user_interaction_data = URM.data[start_user_position:end_user_position]

                # Test interactions
                user_interaction_items_test = user_interaction_items[indices[2]]
                user_interaction_data_test = user_interaction_data[indices[2]]

                mask = user_interaction_data_test > self.test_rating_threshold
                user_interaction_items_test = user_interaction_items_test[mask]
                user_interaction_data_test = user_interaction_data_test[mask]

                URM_test_builder.add_data_lists([user_id] * len(user_interaction_data_test), user_interaction_items_test,
                                                user_interaction_data_test)

                # validation interactions
                if self.with_validation:
                    user_interaction_items_validation = user_interaction_items[indices[1]]
                    user_interaction_data_validation = user_interaction_data[indices[1]]

                    # Remove from validation interactions below a given threshold
                    mask = user_interaction_data_validation > self.test_rating_threshold
                    user_interaction_items_validation = user_interaction_items_validation[mask]
                    user_interaction_data_validation = user_interaction_data_validation[mask]

                    URM_validation_builder.add_data_lists([user_id] * len(user_interaction_data_validation),
                                                          user_interaction_items_validation,
                                                          user_interaction_data_validation)

                # Train interactions
                user_interaction_items_train = user_interaction_items[indices[0]]
                user_interaction_data_train = user_interaction_data[indices[0]]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

            URM_train[URM_name] = URM_train_builder.get_SparseMatrix()
            URM_test[URM_name] = URM_test_builder.get_SparseMatrix()

            if self.with_validation:
                URM_validation[URM_name] = URM_validation_builder.get_SparseMatrix()

        train = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                        postprocessings=dataset.get_postprocessings(),
                        URM_dict=URM_train, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                        ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                        UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        train.remove_users(users_to_remove)

        test = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                       postprocessings=dataset.get_postprocessings(),
                       URM_dict=URM_test, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                       ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                       UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        test.remove_users(users_to_remove)

        if self.with_validation:
            validation = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                                 postprocessings=dataset.get_postprocessings(),
                                 URM_dict=URM_validation, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                                 ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                                 UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
            validation.remove_users(users_to_remove)
            return train, test, validation
        else:
            return train, test



class ColdItemsHoldout(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def __init__(self, train_perc=0.8, validation_perc=0.0, test_perc=0.2, forbid_new_split=False,
                 test_rating_threshold=0, force_new_split=False, allow_cold_users=False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        assert train_perc + validation_perc + test_perc == 1, "DataSplitterHoldout: percentages do not sum to 1"

        super(ColdItemsHoldout, self).__init__(forbid_new_split=forbid_new_split, force_new_split=force_new_split,
                                               allow_cold_users=allow_cold_users, with_validation=validation_perc > 0)
        self.test_rating_threshold = test_rating_threshold
        self.train_perc = train_perc
        self.validation_perc = validation_perc
        self.test_perc = test_perc


    def get_name(self):
        return "cold_items_holdout_{:.2f}_{:.2f}_{:.2f}_testtreshold_{:.1f}{}" \
            .format(self.train_perc, self.validation_perc, self.test_perc, self.test_rating_threshold,
                    "" if self.allow_cold_users else "_no_cold_users")


    def split(self, dataset, random_seed=42):

        super(ColdItemsHoldout, self).split(dataset, random_seed=random_seed)

        n_users, n_items = dataset.n_users, dataset.n_items
        URM_train, URM_test, URM_validation = {}, {}, {}

        items_split = np.random.choice(3, n_items, replace=True, p=[self.train_perc, self.validation_perc, self.test_perc])
        train_items = np.arange(n_items)[items_split == 0]
        validation_items = np.arange(n_items)[items_split == 1]
        test_items = np.arange(n_items)[items_split == 2]

        #Select apriori how to randomizely sort every user
        users_to_remove = []

        for URM_name in dataset.get_URM_names():

            URM = dataset.get_URM(URM_name)
            URM = sps.csr_matrix(URM)

            URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                        auto_create_col_mapper=False, n_cols=n_items)

            URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                       auto_create_col_mapper=False, n_cols=n_items)

            if self.with_validation:
                URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                                 auto_create_col_mapper=False, n_cols=n_items)

            for user_id in range(n_users):

                start_user_position = URM.indptr[user_id]
                end_user_position = URM.indptr[user_id + 1]

                user_interaction_items = URM.indices[start_user_position:end_user_position]
                user_interaction_data = URM.data[start_user_position:end_user_position]

                # Test interactions
                indices = np.in1d(user_interaction_items, test_items, assume_unique=True)
                user_interaction_items_test = user_interaction_items[indices]
                user_interaction_data_test = user_interaction_data[indices]

                # Remove from test interactions below a given threshold
                mask = user_interaction_data_test > self.test_rating_threshold
                user_interaction_items_test = user_interaction_items_test[mask]
                user_interaction_data_test = user_interaction_data_test[mask]

                URM_test_builder.add_data_lists([user_id] * len(user_interaction_data_test),
                                                user_interaction_items_test,
                                                user_interaction_data_test)

                # validation interactions
                if self.with_validation:
                    indices = np.in1d(user_interaction_items, validation_items, assume_unique=True)
                    user_interaction_items_validation = user_interaction_items[indices]
                    user_interaction_data_validation = user_interaction_data[indices]

                    # Remove from validation interactions below a given threshold
                    mask = user_interaction_data_validation > self.test_rating_threshold
                    user_interaction_items_validation = user_interaction_items_validation[mask]
                    user_interaction_data_validation = user_interaction_data_validation[mask]

                    URM_validation_builder.add_data_lists([user_id] * len(user_interaction_data_validation),
                                                          user_interaction_items_validation,
                                                          user_interaction_data_validation)

                    #if len(user_interaction_items_validation) <= 0:
                    #    users_to_remove.append(user_id)

                # Train interactions
                indices = np.in1d(user_interaction_items, train_items, assume_unique=True)
                user_interaction_items_train = user_interaction_items[indices]
                user_interaction_data_train = user_interaction_data[indices]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

                #if len(user_interaction_items_test) <= 0:
                #    users_to_remove.append(user_id)

                if not self.allow_cold_users and len(user_interaction_items_train) <= 0:
                    users_to_remove.append(user_id)

            URM_train[URM_name] = URM_train_builder.get_SparseMatrix()
            URM_test[URM_name] = URM_test_builder.get_SparseMatrix()

            if self.with_validation:
                URM_validation[URM_name] = URM_validation_builder.get_SparseMatrix()

        train = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(), postprocessings=dataset.get_postprocessings(),
                          URM_dict=URM_train, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                          ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                          UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        train.remove_users(users_to_remove)

        test = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(), postprocessings=dataset.get_postprocessings(),
                          URM_dict=URM_test, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                          ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                          UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        test.remove_users(users_to_remove)

        if self.with_validation:
            validation = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                                 postprocessings=dataset.get_postprocessings(),
                                 URM_dict=URM_validation, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                                 ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                                 UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
            validation.remove_users(users_to_remove)
            return train, test, validation
        else:
            return train, test
