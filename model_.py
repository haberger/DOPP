import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from feature_selection import create_traintestsplit

def apply_lassocv(df, target, features, corr_cols, meta_cols, scaler=StandardScaler(), cv=5, random_state=45678, max_iter=100000, fprint=True):

    output = dict()
    
    X_train, X_test, y_train, y_test = create_traintestsplit(df, corr_cols, meta_cols, target_col=target)

    model = make_pipeline(
        scaler, 
        LassoCV(cv=cv,
                random_state=random_state, 
                max_iter=max_iter)
                )
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    output['model'] = model
    output['pred'] = y_pred

    if fprint:
        print(f'current target: {target}')
        print(f'rmse: {rmse(y_test, y_pred, squared=False)}')
        print(f'r2: {r2(y_test, y_pred)}')
        print(f'alpha: {model["lassocv"].alpha_}')
        print()

    output['rmse'] = rmse(y_test, y_pred, squared=False)
    output['r2'] = r2(y_test, y_pred)
    output['alpha'] = model["lassocv"].alpha_

    # Read out attributes
    coeff_df = pd.DataFrame(columns=features, index=[target])
    coeff_df.loc[target] = model['lassocv'].coef_     # dense np.array

    if fprint:
        print('coefficients:')
        display(coeff_df)
        print()
    
    
    coeff_rel_df = (coeff_df.abs().div(coeff_df.abs().sum(axis=1),axis=0))
    if fprint:
        print('importance of features:')
        display((coeff_df.abs().div(coeff_df.abs().sum(axis=1),axis=0)))
        print()

    output['feat_importance'] = coeff_rel_df

    if fprint:
        print('importance of features rank:')
        display(coeff_rel_df.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64'))
        print()

    output['feat_importance_rank'] = coeff_rel_df.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64')

    return output
    
def apply_rf(df, target, features, corr_cols, meta_cols, scaler=StandardScaler(), cv=5, random_state=45678, max_depth=None, max_features=None, fprint=True):
    output = dict()
    
    X_train, X_test, y_train, y_test = create_traintestsplit(df, corr_cols, meta_cols, target_col=target)

    model = make_pipeline(
        scaler, 
        RandomForestRegressor(
            random_state=random_state,
            max_depth=max_depth,
            max_features=max_features
            )
    )

    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    output['model'] = model
    output['pred'] = y_pred
    #output['feat_importance'] = model["randomforestregressor"].feature_importances_
    
    output['rmse'] = rmse(y_test, y_pred, squared=False)
    output['r2'] = r2(y_test, y_pred)

    if fprint:
        print(f'current target: {target}')
        print(f'rmse: {rmse(y_test, y_pred, squared=False)}')
        print(f'r2: {r2(y_test, y_pred)}')
        print(f'fi: {model["randomforestregressor"].feature_importances_}')
        print()

    # gini 
    coeff_gini = pd.DataFrame(columns=features, index=[target])
    coeff_gini.loc[target] = model["randomforestregressor"].feature_importances_
    
    if fprint:
        print('importance of features (gini):')
        display(coeff_gini)
        print()
    
    output['feat_importance'] = coeff_gini

    if fprint:
        print('importance of features rank:')
        display(coeff_gini.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64'))
        print()

    output['feat_importance_rank'] = coeff_gini.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64')

    return output

def apply_gridsearch_rf(df, target, features, param_grid, corr_cols, meta_cols, scaler=StandardScaler(), fprint=True):
    output = dict()

    X_train, X_test, y_train, y_test = create_traintestsplit(df, corr_cols, meta_cols, target_col=target)

    model = make_pipeline(scaler,
        RandomForestRegressor(
            random_state=45678
        )
    )
    reg = GridSearchCV(model, param_grid, scoring="r2", refit=True)
    reg.fit(X_train[list(features)], y_train)

    model = reg.best_estimator_
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    output['model'] = model
    params_df = pd.DataFrame(columns=reg.best_params_.keys(), index=[target])
    params_df.loc[target] = list(reg.best_params_.values())
    output['params'] = params_df
    output['pred'] = y_pred
    # output['feat_importance'] = model["randomforestregressor"].feature_importances_

    output['rmse'] = rmse(y_test, y_pred, squared=False)
    output['r2'] = r2(y_test, y_pred)

    if fprint:
        print(f'current target: {target}')
        print(f'best_parameters: {reg.best_params_}')
        print(f'rmse: {rmse(y_test, y_pred, squared=False)}')
        print(f'r2: {r2(y_test, y_pred)}')
        print(f'fi: {model["randomforestregressor"].feature_importances_}')
        print()

    # gini 
    coeff_gini = pd.DataFrame(columns=features, index=[target])
    coeff_gini.loc[target] = model["randomforestregressor"].feature_importances_
    
    if fprint:
        print('importance of features (gini):')
        display(coeff_gini)
        print()
    
    output['feat_importance'] = coeff_gini

    if fprint:
        print('importance of features rank:')
        display(coeff_gini.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64'))
        print()

    output['feat_importance_rank'] = coeff_gini.replace(0,np.nan).rank(axis=1, ascending=False).astype('Int64')

    return output