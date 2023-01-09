import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def pre_select(X, y, k=20):
    feat_selector = SelectKBest(f_regression, k=k)
    feat_selector.fit(X, y)
    best_feats = feat_selector.get_feature_names_out(X.columns)
    return(best_feats)