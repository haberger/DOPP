import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

def drop_date_columns(df):
    date_columns = [c for c in df.columns if 'date' in c]
    df = df.drop(date_columns, axis=1)
    return df

def create_traintestsplit(df, corr_cols, meta_cols, random_state=424242, target_col='wbgi_cce', reduced=True):
    def divide_into_test_train(df, target, feats_cols, corr_column=target_col):

        x = df.copy()
        x = x[x.cname.isin(target)]
        y = x.loc[:, corr_column]
        x = x.loc[:, feats_cols]
        return x, y

    country_data = pd.DataFrame(df.groupby('cname')['sub-region'].min())
    country_data = country_data.reset_index(drop=False)

    X_train, X_test, y_train, y_test = train_test_split(country_data, country_data['cname'], test_size=0.2, random_state=random_state, stratify=country_data['sub-region'])

    if reduced:

        df.loc[:,['ti_cpi', 'ti_cpi_om']] = df.loc[:,['ti_cpi', 'ti_cpi_om']].replace(np.NaN, -5)
        df_cols_reduced = df.dropna(how='any', axis=1).copy()
        df_cols_reduced.loc[:,['ti_cpi', 'ti_cpi_om']] = df_cols_reduced.loc[:,['ti_cpi', 'ti_cpi_om']].replace(-5, np.NaN)
        df_cols_reduced = df_cols_reduced.dropna(subset = target_col, how='any', axis=0)

        feats_cols_reduced = df_cols_reduced.columns.difference(corr_cols+meta_cols)

        X_train, y_train = divide_into_test_train(df_cols_reduced, y_train, feats_cols_reduced, corr_column=target_col)
        X_test, y_test = divide_into_test_train(df_cols_reduced, y_test, feats_cols_reduced, corr_column=target_col)
    else:
        feats_cols = df.columns.difference(corr_cols+meta_cols)
        X_train, y_train = divide_into_test_train(df, y_train, feats_cols, corr_column=target_col)
        X_test, y_test = divide_into_test_train(df, y_test, feats_cols, corr_column=target_col)

    return X_train, X_test, y_train, y_test


def pre_select(X, y, k=30):
    feat_selector = SelectKBest(f_regression, k=k)
    feat_selector.fit(X, y)
    best_feats = feat_selector.get_feature_names_out(X.columns)

    return best_feats

def filter_highly_correlated(X, feats, corr_target=0.85):

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

    display(highly_corr)

    return feats_reduced

def filter_highly_correlated1(X, feats, tolerance=30):
    print(1)
    corrMatrix = X[feats].corr()
    ct = len(corrMatrix)

    while True:
        ct -= 1
        cols = corrMatrix.keys()
        vals, vecs = np.linalg.eig(corrMatrix, )
        vals1 = (max(vals)/vals)

        print(min(vals1))

        vals1 = np.sqrt(vals1)

        # ll = []
        # for x in vals1:
        #     try:
        #         math.sqrt(x)
        #     except Exception as e:
        #         print(e, x)

        if max(vals1) <= tolerance:
            break

        for i, val in enumerate(vals):
            if val == min(vals):
                for j, vec in enumerate(vecs[:, i]):
                    if abs(vec) == max(abs(vecs[:, i])):
                        mask = np.ones(len(corrMatrix), dtype=bool)
                        for n,col in enumerate(corrMatrix.keys()):
                            mask[n] = n != j
                        corrMatrix = corrMatrix[mask]
                        corrMatrix.pop(cols[j])

    #print(corrMatrix)

    return list(corrMatrix.columns)

    #Feature selection class to eliminate multicollinearity


class MultiCollinearityEliminator():
    
    #Class Constructor
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    #Method to create and return the feature correlation matrix dataframe
    def createCorrMatrix(self, include_target = False):
        #Checking we should include the target in the correlation matrix
        if (include_target == False):
            df_temp = self.df.drop([self.target], axis =1)
            
            #Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
            #Setting min_period to 30 for the sample size to be statistically significant (normal) according to 
            #central limit theorem
            corrMatrix = df_temp.corr(method='pearson', min_periods=30).abs()
        #Target is included for creating the series of feature to target correlation - Please refer the notes under the 
        #print statement to understand why we create the series of feature to target correlation
        elif (include_target == True):
            corrMatrix = self.df.corr(method='pearson', min_periods=30).abs()
        return corrMatrix

    #Method to create and return the feature to target correlation matrix dataframe
    def createCorrMatrixWithTarget(self):
        #After obtaining the list of correlated features, this method will help to view which variables 
        #(in the list of correlated features) are least correlated with the target
        #This way, out the list of correlated features, we can ensure to elimate the feature that is 
        #least correlated with the target
        #This not only helps to sustain the predictive power of the model but also helps in reducing model complexity
        
        #Obtaining the correlation matrix of the dataframe (along with the target)
        corrMatrix = self.createCorrMatrix(include_target = True)                           
        #Creating the required dataframe, then dropping the target row 
        #and sorting by the value of correlation with target (in asceding order)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target]).drop([self.target], axis = 0).sort_values(by = self.target)                    
        return corrWithTarget

    #Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self):
        #Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrMatrix(include_target = False)                          
        colCorr = []
        #Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            #Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():                                            
                if(row[column]>self.threshold) and (row[column]<1):
                    #Adding the features that are not already in the list of correlated features
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        return colCorr

    #Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()                                  
        for idx, row in corrWithTarget.iterrows():
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis =1)
                break
        return self.df

    #Method to run automatically eliminate multicollinearity
    def autoEliminateMulticollinearity(self):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()                                       
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features) 
            #that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()                                     
        return self.df