import pandas as pd
import plotly.express as px

def get_rel_err_df(rf_bf, df, corr_cols, threshold):
    relative_errs = {}
    for target in corr_cols:
        relative_errs[target] = rf_bf[target]['rel_err']
    relative_errs = pd.DataFrame(relative_errs)
    id_cols = ['region', 'sub-region', 'cname', 'year']
    for c in id_cols:
        relative_errs[c] = df.loc[relative_errs.index, c]

    relative_errs = relative_errs.reset_index(drop=False)
    relative_errs = relative_errs.rename({'index': 'original_id'}, axis=1)

    relative_errs = relative_errs.melt(id_vars=id_cols + ['original_id'], var_name='corr_id', value_name='rel_err')
    
    relative_errs.loc[relative_errs.rel_err >= threshold, 'rel_err_status'] = 'bad'
    relative_errs.loc[relative_errs.rel_err < threshold, 'rel_err_status'] = 'good'

    return relative_errs


def get_rel_err_df_stats(df):
    relative_errs_per_country = pd.DataFrame(df.groupby(['region', 'cname', 'corr_id', 'rel_err_status']).rel_err.count())
    relative_errs_per_country_total = pd.DataFrame(df.groupby(['region', 'cname', 'rel_err_status']).rel_err.count())

    relative_errs_per_country_total['corr_id'] = 'total'
    relative_errs_per_country_total.set_index('corr_id', append=True, inplace=True)
    relative_errs_per_country_total = relative_errs_per_country_total.reorder_levels(['region', 'cname', 'corr_id', 'rel_err_status'])
    
    relative_errs_per_country_full = pd.concat([relative_errs_per_country, relative_errs_per_country_total])
    relative_errs_per_country_full = pd.DataFrame(relative_errs_per_country_full.groupby(level=['cname', 'corr_id', 'rel_err_status']).rel_err.sum())
    return relative_errs_per_country_full


def get_plots(relative_errs, q_dict, corr_id):
    plot_dict = {}
    
    dfs = {}
    dfs['normal'] = relative_errs[(relative_errs.rel_err <= q_dict[corr_id]) & (relative_errs.corr_id == corr_id)]
    dfs['outlier'] = relative_errs[(relative_errs.rel_err > q_dict[corr_id]) & (relative_errs.corr_id == corr_id)]
    dfs['full'] = relative_errs[(relative_errs.corr_id == corr_id)]
    
    for key, df in dfs.items():
        plot_dict[f'{key}_dist'] = px.histogram(df, x='rel_err', title=f'Distribution of relative errors for {corr_id}, {key} range')
    
    for key, df in dfs.items():
        plot_dict[f'{key}_dist_region'] = px.histogram(df, x='rel_err', color='region', barmode='overlay', histnorm='probability',
            title=f'Distribution of relative errors for {corr_id} (by region), {key} range')
        
    for key, df in dfs.items():
        plot_dict[f'{key}_dist_sub_region'] = px.histogram(df, x='rel_err', color='sub-region', barmode='overlay', histnorm='probability',
            title=f'Distribution of relative errors for {corr_id} (by sub-region), {key} range')
    
    return plot_dict

