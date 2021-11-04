#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Cesare Bernardis
"""


import numpy as np

from RecSysFramework.DataManager import Dataset
from .DatasetPostprocessing import DatasetPostprocessing


class ImplicitURM(DatasetPostprocessing):

    """
    This class transforms the URM from explicit (or whatever data content it had) to implicit
    """

    def __init__(self, min_rating_threshold=0, keep_original_ratings=False):
        super(ImplicitURM, self).__init__()
        self.min_rating_threshold = min_rating_threshold
        self.keep_original_ratings = keep_original_ratings


    def get_name(self):
        return "implicit_{}{}".format(self.min_rating_threshold, "_or" if self.keep_original_ratings else "")


    def apply(self, dataset):

        new_URM_dict = {}
        for URM_name in dataset.get_URM_names():
            print("DatasetPostprocessing - ImplicitURM: Implicitizing {}".format(URM_name))
            new_URM_dict[URM_name] = dataset.get_URM(URM_name)
            mask = np.ones(new_URM_dict[URM_name].data.size, dtype=np.bool)
            mask[new_URM_dict[URM_name].data > self.min_rating_threshold] = False
            new_URM_dict[URM_name].data[mask] = 0.0
            new_URM_dict[URM_name].eliminate_zeros()
            if not self.keep_original_ratings:
                new_URM_dict[URM_name].data[:] = 1.0

        return Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                       postprocessings=dataset.get_postprocessings() + [self],
                       URM_dict=new_URM_dict, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                       ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                       UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
