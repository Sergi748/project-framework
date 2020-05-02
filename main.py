# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:12:46 2020

@author: sc250091
"""

# Librerias
import os
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
import numpy as np
import pickle
# from scipy.stats import norm
# from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# import xgboost as xgb

# Clases creadas
from tratamientoDatos import tratamiento
from seleccionVariables import SelectVars
from plots import createPlots
from modelosClasificacion import modelos
from predicciones import predictions

class main():
    
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.dfPred = pd.DataFrame()
        self.diccionario = pd.DataFrame()
        self.path = ""
        self.model = ""
        self.nameSave = ""
        self.Id = ""
        self.test_size = int()
        self.nVars = int()
        self.cv = int()
        self.estimator = ""
        self.score_metric = ""
        self.estimatorsVotClass = ""
        self.parametrosLogReg = {}
        self.parametrosRandomForest = {}
        self.parametrosGradBoosting = {}
        self.parametrosvotClass = {}
        self.parametrosxgb = {}

    def createDirectory(self):
        listDir = ['output', 'governance']
        for i in listDir:
            if os.path.exists(self.path + '/' + self.nameModel + '/' + i) == False:
                os.makedirs(self.path + '/' + self.nameModel + '/' + i)
        
        listDirOut = ['modelos', 'metricas', 'predicciones']
        for i in listDirOut:
            if os.path.exists(self.path + '/' + self.nameModel + '/output/' + i) == False:
                os.makedirs(self.path + '/' + self.nameModel + '/output/' + i)
    
    def inputNAs(self):
        self.df = tratamiento().inputNA(self.df, self.path, self.nameModel, self.diccionario)
        
    def dummies(self):
        self.df = tratamiento().createDummies(self.df, self.target, self.path, self.nameModel)    
                          
    def SelectKBest(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectKBest()

    def SelectFromModel(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectFromModel(self.estimator)

    def SelectVariance(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectVariance()
          
    def modelLogisticRegression(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameModel, self.cv, self.score_metric).modelLogisticRegression(self.parametrosLogReg)
    
    def modelRandomForest(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameModel, self.cv, self.score_metric).modelRandomForest(self.parametrosRandomForest)

    def modelGradientBoostingClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameModel, self.cv, self.score_metric).modelGradientBoostingClassifier(self.parametrosGradBoosting)

    def modelVotingClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameModel, self.cv, self.score_metric).modelVotingClassifier(self.estimatorsVotClass, self.parametrosvotClass)

    def modelXGBClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameModel, self.cv, self.score_metric).modelXGBClassifier(self.parametrosxgb)

    def predict(self):
        predictions().predict(self.dfPred, self.path, self.nameModel, self.model, self.nameSave, self.target, self.Id)


finalModel = main(df_f, 'Churn')
finalModel.path = 'C:/Users/SC250091/Documents/SERGIO/Kaggle/Target binaria/Telco_customer_churn'
finalModel.nameModel = 'projectChrun'
finalModel.createDirectory()
finalModel.test_size = 0.2
finalModel.nVars = 25
finalModel.cv = 5
finalModel.score_metric = 'roc_auc'
finalModel.estimator = LogisticRegression()
finalModel.SelectKBest()
finalModel.SelectFromModel()
finalModel.SelectVariance()
finalModel.parametrosLogReg = {"random_state":[22], "max_iter":[100]}
finalModel.parametrosRandomForest = {"random_state":[22], "max_leaf_nodes":[5, 10, 15, 20], "min_samples_split":[5, 10, 30]}
finalModel.parametrosGradBoosting = {'n_estimators':[50, 75, 100], 'random_state':[22], 'max_leaf_nodes':[3,5], 'max_depth':[1, 5, 10]}
finalModel.estimatorsVotClass = VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())
                                                              , ('GradBoost', GradientBoostingClassifier())], n_jobs=5, voting='soft')
finalModel.parametrosvotClass = {'lr__max_iter': [50,100], 'rf__max_leaf_nodes': [5, 10], 'rf__min_samples_split': [5, 10]
                                  , 'GradBoost__n_estimators': [50, 75], 'GradBoost__max_leaf_nodes': [3, 5], 'GradBoost__max_depth': [1, 5]}
finalModel.parametrosxgb = {'random_state':[22], 'max_depth':[1, 5], 'max_leaves':[1, 5, 10], 'n_estimators':[10, 50], 'learning_rate': [0.05, 0.1, 0.5]}
finalModel.modelLogisticRegression()
finalModel.modelRandomForest()
finalModel.modelGradientBoostingClassifier()
finalModel.modelVotingClassifier()
finalModel.modelXGBClassifier()
