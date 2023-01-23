import pandas as pd
import plotly.express as px

def get_rel_err_df(rf_bf, df, corr_cols):
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

    return relative_errs.copy()


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


def get_plots_by_region(relative_errs):
    for corr_id in ['ti_cpi', 'ti_cpi_om', 'bci_bci', 'wbgi_cce']:
        fig = px.box(relative_errs[relative_errs.corr_id == corr_id], 
                        y="rel_err", 
                        x="region", 
                        title=f'Relative error distribution by region for {corr_id}', 
                        hover_data=['corr_id', 'year', 'cname'])
        fig.update_layout(
            width=700,
            height=400,
            margin={'t': 35, 'b': 35}
        )

        fig.show()
        
def get_plots_by_subregion(relative_errs):
    for corr_id in ['ti_cpi', 'ti_cpi_om', 'bci_bci', 'wbgi_cce']:
        fig = px.box(relative_errs[relative_errs.corr_id == corr_id], 
                        x="rel_err", 
                        y="sub-region", 
                        color='region',
                        title=f'Relative error distribution by sub-region for {corr_id}', 
                        hover_data=['corr_id', 'year', 'cname'])
        fig.update_layout(
            width=1000,
            height=500,
            margin={'t': 35, 'b': 35}
        )

        fig.show()

