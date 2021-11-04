#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/05/2021

@author: Cesare Bernardis
"""

from RecSysFramework.DataManager.Reader import Movielens20MReader


class Movielens25MReader(Movielens20MReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
    DATASET_SUBFOLDER = "Movielens25M/"
    ML_VERSION = "ml-25m"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]
