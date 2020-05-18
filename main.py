# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:12:46 2020

@author: sc250091
"""

# Librerias
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Clases creadas
from tratamientoDatos import tratamiento
from seleccionVariables import SelectVars
from modelosClasificacion import modelos
from predicciones import predictions

class main():
    
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.dfPred = pd.DataFrame()
        self.diccionarioNA = pd.DataFrame()
        self.path = ""
        self.nameProject = ""
        self.modelToPred = ""
        self.nameSavePred = ""
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
        self.parametroslgb = {}
        self.parametrosknn = {}

    def createDirectory(self):
        listDir = ['output', 'governance']
        for i in listDir:
            if os.path.exists(self.path + '/' + self.nameProject + '/' + i) == False:
                os.makedirs(self.path + '/' + self.nameProject + '/' + i)
        
        listDirOut = ['modelos', 'metricas', 'predicciones']
        for i in listDirOut:
            if os.path.exists(self.path + '/' + self.nameProject + '/output/' + i) == False:
                os.makedirs(self.path + '/' + self.nameProject + '/output/' + i)
    
    def inputNAs(self):
        self.df = tratamiento().inputNA(self.df, self.path, self.nameProject, self.diccionarioNA)
        
    def dummies(self):
        self.df = tratamiento().createDummies(self.df, self.target, self.path, self.nameProject)    
                          
    def SelectKBest(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectKBest()

    def SelectFromModel(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectFromModel(self.estimator)

    def SelectVariance(self):
        self.df = SelectVars(self.df, self.target, self.nVars).SelectVariance()
          
    def modelLogisticRegression(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelLogisticRegression(self.parametrosLogReg)
    
    def modelRandomForest(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelRandomForest(self.parametrosRandomForest)

    def modelGradientBoostingClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelGradientBoostingClassifier(self.parametrosGradBoosting)

    def modelVotingClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelVotingClassifier(self.estimatorsVotClass, self.parametrosvotClass)

    def modelXGBClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelXGBClassifier(self.parametrosxgb)

    def modelLightGBMClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelLightGBMClassifier(self.parametroslgb)
    
    def modelKNeighborsClassifier(self):
        modelos(self.df, self.target, self.test_size, self.path, self.nameProject, self.cv, self.score_metric).modelKNeighborsClassifier(self.parametrosknn)

    def predict(self):
        predictions().predict(self.dfPred, self.path, self.nameProject, self.modelToPred, self.nameSavePred, self.target, self.Id)

    def plotPredict(self):
        predictions().plotPredict(self.dfPred, self.target, self.nameSavePred, self.path, self.nameProject)
        