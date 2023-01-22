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

def drop_rows_with_nan_values(df, threshold=0.1):
    df = df.dropna(axis='columns', thresh=len(df.index)*threshold)
    return df

def transform_to_categorical(df, threshold=10):
    cols_to_exclude = ['region', 'subregion']
    labels = df.columns[(df.nunique() <= threshold) & (df.nunique() > 2)]
    labels = [l for l in labels if l not in cols_to_exclude]
    df_cat = pd.get_dummies(df, columns=labels)
    return df_cat

def drop_certain_columns(df, 
                    columns_to_remove=(['vdem_corr', 'vdem_execorr', 'vdem_jucorrdc', 'vdem_pubcorr', 
                    'bci_bcistd', 'vdem_exbribe', 'vdem_excrptps', 'vdem_exembez', 'vdem_exthftps', 'vdem_mecorrpt', 
                    'wbgi_ccs', 'wbgi_rle', 'wbgi_rln', 'wbgi_rls', 'bci_bcistd', 'vdem_exbribe', 'vdem_excrptps', 
                    'vdem_exembez', 'vdem_exthftps', 'vdem_mecorrpt', 'wbgi_ccs'])):

    df = df.drop(columns=columns_to_remove)
    return df

def load_reduced_df(corr_cols):
    data_dir = 'data'
    qog_dataset_filename = 'qog_std_ts_jan22.csv'
    df = pd.read_csv(join(data_dir, qog_dataset_filename), low_memory=False)

    df = merge_region(df)
    df_reduced = drop_certain_columns(df)
    df_reduced = return_rows_where_all_corruption_data_is_available(df, corr_cols)
    df_reduced = drop_rows_with_nan_values(df_reduced)
    df_reduced = transform_to_categorical(df_reduced)
    return df_reduced