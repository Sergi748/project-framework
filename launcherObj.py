# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:05:18 2020

@author: Sergio Campos
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from main import main 
df = pd.read_csv('C:/Users/SC250091/OneDrive - Teradata/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn/01_Datos/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')
df.Churn = np.where(df.Churn == 'Yes', 1, 0)
dfPred = pd.read_csv('C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn/01_Datos/dfPred.csv', sep=';')
dfPredReal = pd.read_csv('C:/Users/sc250091/OneDrive - Teradata/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn/01_Datos/dfPred.csv', sep=';')[['customerID', 'Churn']]


finalModelComplete = main(df, 'Churn')
# Parameters
finalModelComplete.path = 'C:/Users/SC250091/OneDrive - Teradata/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn'
finalModelComplete.nameProject = 'projectChurn_2'
finalModelComplete.Id = 'customerID'
finalModelComplete.dfPred = dfPred
finalModelComplete.modelToPred = 'LGBMClassifier'
finalModelComplete.nameSavePred = 'Prueba_LGB'
finalModelComplete.test_size = 0.2
finalModelComplete.nVars = 25
finalModelComplete.cv = 5
finalModelComplete.score_metric = 'roc_auc'
finalModelComplete.scaler = 'StandardScaler'

# Directory and training
finalModelComplete.createDirectory()
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
finalModelComplete.parametroslgb = {'random_state':[22], 'max_depth':[1, 5], 'max_leaves':[1, 5, 10], 'n_estimators':[10, 50], 'learning_rate': [0.05, 0.1, 0.5]}
finalModelComplete.modelLogisticRegression()
finalModelComplete.modelRandomForest()
finalModelComplete.modelGradientBoostingClassifier()
finalModelComplete.modelVotingClassifier()
finalModelComplete.modelXGBClassifier()
finalModelComplete.modelLightGBMClassifier()

# Prediction
finalModelComplete.predictClassifier()
finalModelComplete.dfPredReal = dfPredReal
finalModelComplete.plotPredictClassifier()
