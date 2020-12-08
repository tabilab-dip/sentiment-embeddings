#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
import argparse
import os
import sys
import warnings

import constants
import pickle
import preprocessing
import svm

warnings.filterwarnings("ignore")
constants.LANG = "turkish"
constants.EMBEDDING_TYPE = "ensemble"
constants.EMBEDDING_SIZE = 100
constants.CV_NUMBER = 10
constants.USE_3_REV_POL_SCORES = True
constants.DATASET_PATH = "input/Sentiment_dataset_turk.csv"   
constants.DISAMB_PATH = "sentiment.disamb.txt"
constants.MODEL_FILE_NAME = "finalized_model.sav"


pre = preprocessing.Preprocessing()
(model, sf, tr_vecs, imp) = pickle.load(open(constants.MODEL_FILE_NAME, "rb"))

def evaluate(text):
    lines = [line for line in text.split("\n") if line]
    for line in lines:
        line = line.strip("\n")
        line = pre.preprocess_one_line(line)
        sentiment = svm.test_model(model, sf, tr_vecs, imp, line)[0][0]
        sentiment = "Positive" if sentiment == "P" else "Negative"
        return sentiment

