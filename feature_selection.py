import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

def drop_date_columns(df):
    date_columns = [c for c in df.columns if 'date' in c]
    df = df.drop(date_columns, axis=1)
    return df

def create_traintestsplit(df, random_state=424242, target_col='wbgi_cce', reduced=True):

    def divide_into_test_train(df, target, feats_cols, corr_column=target_col):
        x = df.copy()
        x = x[x.cname.isin(target)]
        y = x.loc[:, corr_column]
        x = x.loc[:, feats_cols]
        return x, y

    country_data = pd.DataFrame(df.groupby('cname')['sub-region'].min())
    country_data = country_data.reset_index(drop=False)

    X_train, X_test, y_train, y_test = train_test_split(country_data, country_data['cname'], test_size=0.2, random_state=random_state, stratify=country_data['sub-region'])

    corr_cols = ['bci_bci', 'ti_cpi', 'vdem_corr', 'vdem_execorr', 'vdem_jucorrdc', 'vdem_pubcorr', 'wbgi_cce']
    if reduced:
        feat_col_start_reduced = 9
        df_cols_reduced = df.dropna(how='any', axis=1)
        feats_cols_reduced = [c for c in df_cols_reduced.columns[feat_col_start_reduced:-2] if c not in corr_cols]
        X_train, y_train = divide_into_test_train(df_cols_reduced, y_train, feats_cols_reduced, corr_column=target_col)
        X_test, y_test = divide_into_test_train(df_cols_reduced, y_test, feats_cols_reduced, corr_column=target_col)
    else:
        feat_col_start_full = 10
        feats_cols = [c for c in df.columns[feat_col_start_full:-2] if c not in corr_cols]
        X_train, y_train = divide_into_test_train(df, y_train, feats_cols, corr_column=target_col)
        X_test, y_test = divide_into_test_train(df, y_test, feats_cols, corr_column=target_col)

    return X_train, X_test, y_train, y_test


def pre_select(X, y, k=20):
    feat_selector = SelectKBest(f_regression, k=k)
    feat_selector.fit(X, y)
    best_feats = feat_selector.get_feature_names_out(X.columns)

    return best_feats


def filter_corruption(feats, corruption_cols = ['bci_bcistd', 'vdem_exbribe', 'vdem_excrptps', 'vdem_exembez', 'vdem_exthftps', 'vdem_mecorrpt', 'wbgi_ccs']):
    feats_filtered = [f for f in feats if f not in corruption_cols]

    return feats_filtered


def filter_highly_correleated(X, feats, corr_target=0.85):

    cm = X[feats].corr()
    cm = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    cm = cm.reset_index(drop=False)
    cm = cm.melt(id_vars='index', var_name='second_col', value_name='corr')
    cm.columns = ['first_col', 'second_col', 'corr_id']
    cm = cm[cm.first_col != cm.second_col]
    cm = cm.dropna()

    highly_corr = cm[cm.corr_id.abs() > corr_target].copy()

    highly_corr.loc[:, 'corr_id'] = highly_corr.corr_id.abs()
    highly_corr_agg = highly_corr.groupby('first_col').agg({'second_col': 'count', 'corr_id': 'mean'})

    # print(highly_corr_agg)

    feats_reduced = [f for f in feats if f not in highly_corr_agg[highly_corr_agg.second_col > 1].index.values]

    return feats_reduced