#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Cesare Bernardis
"""


import numpy as np

from .DatasetPostprocessing import DatasetPostprocessing


class ItemSample(DatasetPostprocessing):

    """
    This class selects a partition of URM such that only some of the original items are present
    """


    def __init__(self, item_quota = 1.0):

        assert item_quota > 0.0 and item_quota <= 1.0,\
            "DataReaderPostprocessing - Item sample: item_quota must be a positive value > 0.0 and <= 1.0, " \
            "provided value was {}".format(item_quota)

        super(ItemSample, self).__init__()
        self.item_quota = item_quota


    def get_name(self):
        return "item_sample_{}".format(self.item_quota)


    def apply(self, dataset):

        print("DatasetPostprocessing - ItemSample: Sampling {:.2f}% of items".format(self.item_quota * 100))
        items_to_remove = np.random.choice(dataset.n_items,
                                           int(dataset.n_items * (1. - self.item_quota)),
                                           replace=False)

        new_dataset = dataset.copy()
        new_dataset.remove_items(items_to_remove, keep_original_shape=False)
        new_dataset.add_postprocessing(self)

        return new_dataset
