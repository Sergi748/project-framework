# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:51:49 2020

@author: sc250091
"""

import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
import numpy as np
import pickle
# from scipy.stats import norm
# from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from plots import createPlots

# Clases creadas
from tratamientoDatos import tratamiento

class predictions():

    def predict(self, df, path, nameModel, model, nameSave, target, Id):
        
        '''Funcion para realizar predicciones'''        
        df1 = df.copy()
        varsTrain = pd.read_csv(path + '/' + nameModel + '/governance/varsTrain.csv', sep=';')
        fileName = path + '/' + nameModel + '/output/modelos/model_' + model + '.pkl'
        loaded_model = pickle.load(open(fileName, 'rb'))
        
        '''Imputamos los NA´s, creamos las dummies correspondientes y seleccionamos las variables del train'''
        df1 = tratamiento().inputNAPred(df1, path, nameModel)
        df1 = tratamiento().createDummiesPred(df1, target, Id)
        
        '''Creamos las variables que nos falte para tener las mismas que en train'''
        if all(elem in df1.columns for elem in varsTrain.Variables_used.tolist()) == False:
            varsToCreate = np.setdiff1d(varsTrain.Variables_used.tolist(), df1.columns)
            for var in varsToCreate:
                df[var] = 0
    
        df1 = df1.loc[:,df1.columns.isin(varsTrain.Variables_used.tolist())]
        df1['score'] = loaded_model.predict_proba(df1)[::,1]
        df1.to_csv(path + '/' + nameModel + '/predicciones/prediccion_model_' + model + '_' + nameSave + '.csv', sep=';', index=False)
