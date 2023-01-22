from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def merge_region(df, df_ccode_column_label='ccode', path_to_region_dataset = 'data/region_info.csv'):
    '''
    df: df that should be enriched with region_data
    df_ccode_column_label: label of column in df that holds the ISO-3166 countrycode
    path_to_region_dataset: path to this dataset: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
    '''
    df_region = pd.read_csv(path_to_region_dataset, low_memory=False)
    df_region.rename(columns={'country-code': df_ccode_column_label}, inplace=True)

    df_merged = df.join(df_region[[df_ccode_column_label, 'region', 'sub-region']].set_index(df_ccode_column_label), on = df_ccode_column_label, how='left')
    df_region.rename(columns={df_ccode_column_label: 'ccode'}, inplace=True)
    # Manual patching of missing regions
    # Czechoslovakia
    df_merged.loc[df_merged.ccode == 200, 'region'] = 'Europe'
    df_merged.loc[df_merged.ccode == 200, 'sub-region'] = 'Eastern Europe'
    # Ethiopia
    df_merged.loc[df_merged.ccode == 230, 'region'] = 'Africa'
    df_merged.loc[df_merged.ccode == 230, 'sub-region'] = 'Sub-Saharan Africa'
    # Germany
    df_merged.loc[(df_merged.ccode == 278) | (df_merged.ccode == 280), 'region'] = 'Europe'
    df_merged.loc[(df_merged.ccode == 278) | (df_merged.ccode == 280), 'sub-region'] = 'Western Europe'
    # Yemen
    df_merged.loc[(df_merged.ccode == 720) | (df_merged.ccode == 886), 'region'] = 'Asia'
    df_merged.loc[(df_merged.ccode == 720) | (df_merged.ccode == 886), 'sub-region'] = 'Western Asia'
    # USSR
    df_merged.loc[df_merged.ccode == 810, 'region'] = 'Europe'
    df_merged.loc[df_merged.ccode == 810, 'sub-region'] = 'Eastern Europe'
    # Yugoslavia and serbia
    df_merged.loc[df_merged.ccode == 891, 'region'] = 'Europe'
    df_merged.loc[df_merged.ccode == 891, 'sub-region'] = 'Southern Europe'
    # Tibet
    df_merged.loc[df_merged.ccode == 994, 'region'] = 'Asia'
    df_merged.loc[df_merged.ccode == 994, 'sub-region'] = 'Eastern Asia'
    # Vietnam North and South
    df_merged.loc[df_merged.ccode == 998, 'region'] = 'Asia'
    df_merged.loc[df_merged.ccode == 998, 'sub-region'] = 'South-eastern Asia'
    df_merged.loc[df_merged.ccode == 999, 'region'] = 'Asia'
    df_merged.loc[df_merged.ccode == 999, 'sub-region'] = 'South-eastern Asia'

    # merge Melanesia/Micronesia/Polynesia to Pacific Islands due to lack of data
    df_merged.loc[df_merged['sub-region']=='Melanesia', 'sub-region'] = 'Pacific Islands'
    df_merged.loc[df_merged['sub-region']=='Micronesia', 'sub-region'] = 'Pacific Islands'
    df_merged.loc[df_merged['sub-region']=='Polynesia', 'sub-region'] = 'Pacific Islands'

    return df_merged


if __name__ == '__main__':
    data_dir = 'data'
    qog_dataset_filename = 'qog_std_ts_jan22.csv'
    df = pd.read_csv(join(data_dir, qog_dataset_filename), low_memory=False)

    df_merged = merge_region(df)
    number_of_na_regions = df_merged.region.isna().sum()
    assert number_of_na_regions == 0, f'Manual patching for some countries required, {number_of_na_regions = }'

    print(df_merged)







