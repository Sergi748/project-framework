# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:05:18 2020

@author: sc250091
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from main import main 


finalModelComplete = main(df, 'Churn')
finalModelComplete.path = 'C:/Users/SC250091/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn'
finalModelComplete.nameModel = 'projectChrun'
finalModelComplete.test_size = 0.2
finalModelComplete.nVars = 25
finalModelComplete.cv = 5
finalModelComplete.score_metric = 'roc_auc'
finalModelComplete.inputNAs()
finalModelComplete.dummies()
finalModelComplete.SelectVariance()
finalModelComplete.parametrosLogReg = {"random_state":[22], "max_iter":[100]}
finalModelComplete.parametrosRandomForest = {"random_state":[22], "max_leaf_nodes":[5, 10, 15, 20], "min_samples_split":[5, 10, 30]}
finalModelComplete.parametrosGradBoosting = {'n_estimators':[50, 75, 100], 'random_state':[22], 'max_leaf_nodes':[3,5], 'max_depth':[1, 5, 10]}
finalModelComplete.estimatorsVotClass = VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())
                                                              , ('GradBoost', GradientBoostingClassifier())], n_jobs=5, voting='soft')
finalModelComplete.parametrosvotClass = {'lr__max_iter': [50,100], 'rf__max_leaf_nodes': [5, 10], 'rf__min_samples_split': [5, 10]
                                  , 'GradBoost__n_estimators': [50, 75], 'GradBoost__max_leaf_nodes': [3, 5], 'GradBoost__max_depth': [1, 5]}
finalModelComplete.parametrosxgb = {'random_state':[22], 'max_depth':[1, 5], 'max_leaves':[1, 5, 10], 'n_estimators':[10, 50], 'learning_rate': [0.05, 0.1, 0.5]}
finalModelComplete.modelLogisticRegression()
finalModelComplete.modelRandomForest()
finalModelComplete.modelGradientBoostingClassifier()
finalModelComplete.modelVotingClassifier()
finalModelComplete.modelXGBClassifier()
finalModelComplete.dfPred = dfPred
finalModelComplete.modelToPred = 'LogisticRegression'
finalModelComplete.nameSavePred = 'Prueba_LogReg'
finalModelComplete.Id = 'customerID'
finalModelComplete.predict()
finalModelComplete.plotPredict()
