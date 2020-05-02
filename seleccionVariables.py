# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:33:01 2020

@author: sc250091
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


class SelectVars():
    
    def __init__(self, df, target, nVars):
        self.df = df
        self.target = target
        self.nVars = nVars
    
    def __split(self):
        return self.df.loc[:, ~self.df.columns.isin([self.target])], self.df.loc[:, self.df.columns == self.target]
          
    def SelectKBest(self):
        df_x, df_y = SelectVars.__split(self)
        n = df_x.shape[1] if not self.nVars else int(self.nVars)
        # Metodo selectKBest con chi2
        X_new = SelectKBest(chi2, k = n).fit(df_x, df_y)
        dfSelectKBestChi2 = pd.DataFrame({'Specs':df_x.columns, 'CoefSelectKBest': X_new.scores_}).sort_values('CoefSelectKBest', ascending=False).reset_index(drop=True)
        varsSelectKBestChi2 = dfSelectKBestChi2.loc[0:n,['Specs']]['Specs'].tolist()
        return self.df.loc[:,varsSelectKBestChi2 + [self.target]]
        
    def SelectFromModel(self, estimator):
        df_x, df_y = SelectVars.__split(self)
        n = df_x.shape[1] if not self.nVars else int(self.nVars)
        # Seleccion de variables basado en un modelo
        model = SelectFromModel(estimator=estimator).fit(df_x, df_y)
        dfSelectFromModel = pd.DataFrame({'Specs':df_x.columns, 'CoefSelectModel':model.estimator_.coef_[0]})
        # Reordenamos los coeficientes en valor absoluto
        dfSelectFromModel = dfSelectFromModel.iloc[(-np.abs(dfSelectFromModel['CoefSelectModel'].values)).argsort()].reset_index(drop=True)
        varsSelectFromModel =  dfSelectFromModel.loc[0:n,['Specs']]['Specs'].tolist()
        return self.df.loc[:,varsSelectFromModel + [self.target]]

    def SelectVariance(self):
        df_x, df_y = SelectVars.__split(self)
        n = df_x.shape[1] if not self.nVars else int(self.nVars)
        # Seleccion de variables por Variance
        selector = VarianceThreshold().fit(df_x, df_y)
        dfSelectVariance = pd.DataFrame({'Specs':df_x.columns, 'CoefSelectVariances': selector.variances_}).sort_values('CoefSelectVariances', ascending=False).reset_index(drop=True)
        varsSelectVariance = dfSelectVariance.loc[0:n,['Specs']]['Specs'].tolist()
        return self.df.loc[:,varsSelectVariance + [self.target]]       




