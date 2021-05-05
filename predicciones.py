# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:51:49 2020

@author: Sergio Campos
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

# Clases creadas
from tratamientoDatos import tratamiento

class predictions():

    def predictClassifier(self, dfPred, path, nameProject, modelToPred, nameSavePred, target, Id):
        
        '''Funcion para realizar predicciones'''        
        df1 = dfPred.copy()
        df1Id = df1[[Id]]
        varsTrain = pd.read_csv(path + '/' + nameProject + '/governance/varsTrain.csv', sep=';')
        fileName = path + '/' + nameProject + '/output/modelos/model_' + modelToPred + '.pkl'
        loaded_model = pickle.load(open(fileName, 'rb'))
    
        '''Imputamos los NA´s, creamos las dummies correspondientes y seleccionamos las variables del train'''
        df1 = tratamiento().inputNAPred(df1, path, nameProject)
        df1 = tratamiento().createDummiesPred(df1, target, Id)
        
        '''Creamos las variables que nos falte para tener las mismas que en train'''
        if all(elem in df1.columns for elem in varsTrain.Variables_used.tolist()) == False:
            varsToCreate = np.setdiff1d(varsTrain.Variables_used.tolist(), df1.columns)
            for var in varsToCreate:
                df1[var] = 0
    
        pathScaler = path + '/' + nameProject + '/output/modelos/scaler_' + modelToPred + '.pkl'
        df1 = df1.loc[:,varsTrain.Variables_used.tolist()]
        if os.path.exists(pathScaler):
            scaler = joblib.load(pathScaler)
            df1_columns = df1.columns
            df1 = pd.DataFrame(scaler.transform(df1), index=df1.index, columns=df1_columns)

        df1['score'] = loaded_model.predict_proba(df1)[::,1]
        df1[Id] = df1Id
        df1[[Id, 'score']].to_csv(path + '/' + nameProject + '/output/predicciones/prediccion_model_' + modelToPred + '_' + nameSavePred + '.csv', sep=';', index=False)

