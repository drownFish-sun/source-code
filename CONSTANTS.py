import os
import pandas as pd
import numpy as np
import math
from numpy.linalg import inv
from scipy import sparse
import argparse
import ast
import os.path
import re
from scipy.spatial import distance
from numpy.linalg import inv
from sklearn.preprocessing import minmax_scale

PROJECT_ROOT = os.path.abspath(os.getcwd())
replace = {}
choices=["Word2vec", "BERT", "GloVe", "TF-IDF"]
def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    try:
        TP = confusion.loc[1, 1]
    except:
        TP = 0
    try:
        TN = confusion.loc[0, 0]
    except:
        TN = 0
    try:
        FP = confusion.loc[0, 1]
    except:
        FP = 0
    try:
        FN = confusion.loc[1, 0]
    except:
        FN = 0

    print("TP={}, TN={}, FP={}, FN={}".format(TP, TN, FP, FN))

    out = {}
    out['ACR'] = round((TP + TN) / (TP + TN + FP + FN) * 100, 4)
    try:
        out['PRE'] = round(TP / (TP + FP) * 100, 4)
    except ZeroDivisionError:
        out['PRE'] = 0
    try:
        out['PF'] = round(FP / (FP + TN) * 100, 4)
    except ZeroDivisionError:
        out['PF'] = 0
    try:
        out['PD'] = round(TP / (TP + FN) * 100, 4)
    except ZeroDivisionError:
        out['PD'] = 0
    try:
        out['F1'] = round((2 * out['PRE'] * out['PD']) / (out['PRE'] + out['PD']), 4)
    except ZeroDivisionError:
        out['F1'] = 0
    print('Acc={}, PRE={}, Recall={}, F1={}'.format(out['ACR'], out['PRE'], out['PD'], out['F1']))
    # out['DI'] = round((TP / (TP + FN)) / (TN / (TN + FP)) * 100, 4)
    # out['TPR'] = round(TP / (TP + FN) * 100, 4)
    # out['TNR'] = round(TN / (FP + TN) * 100, 4)
    return out
def calculate_variance(data):
    n = len(data)
    if n == 0:
        return 0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance