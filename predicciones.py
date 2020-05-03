# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:51:49 2020

@author: sc250091
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Clases creadas
from tratamientoDatos import tratamiento
from plots import createPlots

class predictions():

    def predict(self, dfPred, path, nameModel, modelToPred, nameSavePred, target, Id):
        
        '''Funcion para realizar predicciones'''        
        df1 = dfPred.copy()
        df1Id = df1[[Id]]
        varsTrain = pd.read_csv(path + '/' + nameModel + '/governance/varsTrain.csv', sep=';')
        fileName = path + '/' + nameModel + '/output/modelos/model_' + modelToPred + '.pkl'
        loaded_model = pickle.load(open(fileName, 'rb'))
        
        '''Imputamos los NA´s, creamos las dummies correspondientes y seleccionamos las variables del train'''
        df1 = tratamiento().inputNAPred(df1, path, nameModel)
        df1 = tratamiento().createDummiesPred(df1, target, Id)
        
        '''Creamos las variables que nos falte para tener las mismas que en train'''
        if all(elem in df1.columns for elem in varsTrain.Variables_used.tolist()) == False:
            varsToCreate = np.setdiff1d(varsTrain.Variables_used.tolist(), df1.columns)
            for var in varsToCreate:
                df1[var] = 0
    
        df1 = df1.loc[:,df1.columns.isin(varsTrain.Variables_used.tolist())]
        df1['score'] = loaded_model.predict_proba(df1)[::,1]
        df1[Id] = df1Id
        df1[[Id, 'score']].to_csv(path + '/' + nameModel + '/output/predicciones/prediccion_model_' + modelToPred + '_' + nameSavePred + '.csv', sep=';', index=False)

    def plotPredict(self, df, target, nameSavePred, path, nameModel):
        
        df1 = df.copy()
        df1[target] = df1[target].astype(int)
        dfLiftPred = createPlots().tableLift(df1, 'score', target)
        createPlots().plotROC_AUC_pred(df1, target, nameSavePred, path, nameModel)
        createPlots().plotLift(dfLiftPred, 'predict_' + nameSavePred, path, nameModel)
        
