#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Cesare Bernardis
"""


import numpy as np

from .DatasetPostprocessing import DatasetPostprocessing


class UserSample(DatasetPostprocessing):

    """
    This class selects a partition of URM such that only some of the original users are present
    """


    def __init__(self, user_quota = 1.0):

        assert user_quota > 0.0 and user_quota <= 1.0,\
            "DataReaderPostprocessing - User sample: user_quota must be a positive value > 0.0 and <= 1.0, " \
            "provided value was {}".format(user_quota)

        super(UserSample, self).__init__()
        self.user_quota = user_quota


    def get_name(self):
        return "user_sample_{}".format(self.user_quota)


    def apply(self, dataset):

        print("DatasetPostprocessing - UserSample: Sampling {:.2f}% of users".format(self.user_quota * 100))
        users_to_remove = np.random.choice(dataset.n_users, int(dataset.n_users * (1. - self.user_quota)), replace=False)

        new_dataset = dataset.copy()
        new_dataset.remove_users(users_to_remove, keep_original_shape=False)
        new_dataset.add_postprocessing(self)

        return new_dataset
