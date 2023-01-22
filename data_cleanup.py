import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from region_merge import merge_region

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


def load_reduced_df(corr_cols):
    data_dir = 'data'
    qog_dataset_filename = 'qog_std_ts_jan22.csv'
    df = pd.read_csv(join(data_dir, qog_dataset_filename), low_memory=False)

    df = merge_region(df)
    df_reduced = return_rows_where_all_corruption_data_is_available(df, corr_cols)
    return df_reduced