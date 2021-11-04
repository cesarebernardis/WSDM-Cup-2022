#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""


class DatasetPostprocessing(object):

    """
    This class provides the interface for the DataReaderPostprocessing objects
    """

    def get_name(self):
        pass

    def apply(self, dataset):
        pass

