#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Cesare Bernardis
"""

import numpy as np

from .DatasetPostprocessing import DatasetPostprocessing


class URMKCore(DatasetPostprocessing):

    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """

    def __init__(self, user_k_core, item_k_core, reshape=True):

        assert user_k_core >= 1,\
            "DatasetPostprocessingKCore: user_k_core must be a positive value >= 1, provided value was {}".format(user_k_core)

        assert item_k_core >= 1,\
            "DatasetPostprocessingKCore: item_k_core must be a positive value >= 1, provided value was {}".format(item_k_core)

        super(URMKCore, self).__init__()
        self.user_k_core = user_k_core
        self.item_k_core = item_k_core
        self.reshape = reshape


    def get_name(self):
        return "kcore_user_{}_item_{}{}".format(self.user_k_core, self.item_k_core, "_reshaped" if self.reshape else "")


    def apply(self, dataset):

        from RecSysFramework.DataManager.Utils import select_asymmetric_k_cores

        print("DatasetPostprocessing - KCore: Applying {} user core - {} item core"
              .format(self.user_k_core, self.item_k_core))

        _, removedUsers, removedItems = select_asymmetric_k_cores(dataset.get_URM(), user_k_value=self.user_k_core,
                                                                  item_k_value=self.item_k_core, reshape=False)

        new_dataset = dataset.copy()
        new_dataset.remove_items(removedItems, keep_original_shape=not self.reshape)
        new_dataset.remove_users(removedUsers, keep_original_shape=not self.reshape)
        new_dataset.add_postprocessing(self)

        return new_dataset


class ICMKCore(DatasetPostprocessing):

    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """

    def __init__(self, item_k_core, feature_k_core, reshape=True):

        assert feature_k_core >= 1,\
            "DatasetPostprocessingICMKCore: feature_k_core must be a positive value >= 1, provided value was {}".format(feature_k_core)

        assert item_k_core >= 1,\
            "DatasetPostprocessingICMKCore: item_k_core must be a positive value >= 1, provided value was {}".format(item_k_core)

        super(ICMKCore, self).__init__()
        self.feature_k_core = feature_k_core
        self.item_k_core = item_k_core
        self.reshape = reshape


    def get_name(self):
        return "kcore_item_{}_feature_{}{}".format(self.item_k_core, self.feature_k_core, "_reshaped" if self.reshape else "")


    def apply(self, dataset):

        new_dataset = dataset.copy()
        old_features_number = 0
        new_features_number = new_dataset.get_ICM().shape[1]

        while old_features_number != new_features_number:

            old_features_number = new_features_number
            new_dataset.remove_item_features(min_occurrence=self.feature_k_core, max_perc_occurrence=1.0,
                                             keep_original_shape=not self.reshape)

            ICM_all = new_dataset.get_ICM()

            removeItems = np.arange(ICM_all.shape[0])[np.ediff1d(ICM_all.tocsr().indptr) < self.item_k_core]
            new_dataset.remove_items(removeItems, keep_original_shape=not self.reshape)
            new_features_number = ICM_all.shape[1]

            assert new_features_number > 0, "{}: all features removed, reduce the core constraint"

        new_dataset.add_postprocessing(self)

        return new_dataset



class CombinedKCore(DatasetPostprocessing):

    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """

    def __init__(self, user_k_core, item_urm_k_core, item_icm_k_core, feature_k_core, reshape=True):

        assert user_k_core >= 1,\
            "DatasetPostprocessingURMKCore: user_k_core must be a positive value >= 1, provided value was {}".format(user_k_core)

        assert item_urm_k_core >= 1,\
            "DatasetPostprocessingURMKCore: item_urm_k_core must be a positive value >= 1, provided value was {}".format(item_urm_k_core)

        assert item_icm_k_core >= 1,\
            "DatasetPostprocessingURMKCore: item_icm_k_core must be a positive value >= 1, provided value was {}".format(item_icm_k_core)

        assert feature_k_core >= 1,\
            "DatasetPostprocessingURMKCore: feature_k_core must be a positive value >= 1, provided value was {}".format(feature_k_core)

        super(CombinedKCore, self).__init__()
        self.user_k_core = user_k_core
        self.item_urm_k_core = item_urm_k_core
        self.item_icm_k_core = item_icm_k_core
        self.feature_k_core = feature_k_core
        self.reshape = reshape


    def get_name(self):
        return "kcore_user_{}_item_{}_{}_feature_{}{}".format(self.user_k_core, self.item_urm_k_core,
                                                             self.item_icm_k_core, self.feature_k_core,
                                                             "_reshaped" if self.reshape else "")


    def apply(self, dataset):

        from RecSysFramework.DataManager.Utils import select_asymmetric_k_cores

        new_dataset = dataset.copy()
        old_shape = (0, 0)
        new_shape = new_dataset.get_ICM().shape

        while old_shape != new_shape:

            _, removedUsers, removedItems = select_asymmetric_k_cores(new_dataset.get_URM(),
                                                                      user_k_value=self.user_k_core,
                                                                      item_k_value=self.item_urm_k_core,
                                                                      reshape=False)

            new_dataset.remove_items(removedItems, keep_original_shape=not self.reshape)
            new_dataset.remove_users(removedUsers, keep_original_shape=not self.reshape)

            old_shape = new_dataset.get_ICM().shape

            new_dataset.remove_item_features(min_occurrence=self.feature_k_core, max_perc_occurrence=0.3,
                                             keep_original_shape=not self.reshape)

            ICM_all = new_dataset.get_ICM()

            removeItems = np.arange(ICM_all.shape[0])[np.ediff1d(ICM_all.tocsr().indptr) < self.item_icm_k_core]
            new_dataset.remove_items(removeItems, keep_original_shape=not self.reshape)
            new_shape = new_dataset.get_ICM().shape

            assert new_shape[1] > 0, "{}: all features removed, reduce the core constraint"

        new_dataset.add_postprocessing(self)

        return new_dataset
