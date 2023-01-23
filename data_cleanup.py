import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from region_merge import merge_region
from sklearn.preprocessing import OneHotEncoder

def return_rows_where_all_corruption_data_is_available(df, corr_cols):
    '''BE AWARE THAT ONLY ONE OF ti_cpi and ti_cpi_om IS AVAILABLE NEVER BOTH'''
    merge = corr_cols.copy()
    if 'ti_cpi' in corr_cols and 'ti_cpi_om' in corr_cols:
        merge.remove('ti_cpi') 
        merge.remove('ti_cpi_om') 
        merge.append('merge')
        df['merge'] = df['ti_cpi'].combine_first(df['ti_cpi_om'])
    
    df = df.dropna(subset=merge, axis=0, how='any')

    if 'merge' in df.columns:
        df = df.drop(columns=['merge'])


    return df

def drop_columns_with_nan_values(df, threshold=0.1):
    df = df.dropna(axis='columns', thresh=len(df.index)*threshold)
    return df

def transform_to_categorical(df, threshold=10):
    cols_to_exclude = ['region', 'subregion']
    labels = df.columns[(df.nunique() <= threshold) & (df.nunique() > 2)]
    labels = [l for l in labels if l not in cols_to_exclude]
    df_cat = pd.get_dummies(df, columns=labels, dummy_na=True)
    return df_cat

def drop_certain_columns(df, 
                    corr_cols,
                    meta_cols,
                    columns_to_remove=(['ht_region'])):

    # drop variables that start with those letters
    l = ['wbgi', 'ti', 'bci', 'vdem']
    clist = []

    for c in df.columns:
        if c in meta_cols+corr_cols:
            clist.append(c)
        elif c.split('_')[0] not in l:
            clist.append(c)

    df_red = df[clist].copy()

    df_red = df_red.drop(columns=columns_to_remove)
    return df_red
