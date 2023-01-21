import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from region_merge import merge_region

def return_rows_where_all_corruption_data_is_available(df):
    '''BE AWARE THAT ONLY ONE OF ti_cpi and ti_cpi_om IS AVAILABLE NEVER BOTH'''

    corruption_col = ['bci_bci', 'ti_cpi', 'vdem_corr', 'vdem_execorr', 'vdem_jucorrdc', 'vdem_pubcorr', 'wbgi_cce']

    df_cpi_combined = df.copy()
    df_cpi_combined['ti_cpi']=df['ti_cpi'].combine_first(df['ti_cpi_om'])
    df_all_corruption_info_available = df_cpi_combined.dropna(subset=corruption_col, axis=0, how='any')
    return df_all_corruption_info_available

def load_reduced_df():
    data_dir = 'data'
    qog_dataset_filename = 'qog_std_ts_jan22.csv'
    df = pd.read_csv(join(data_dir, qog_dataset_filename), low_memory=False)

    df = merge_region(df)
    df_reduced = return_rows_where_all_corruption_data_is_available(df)
    return df_reduced