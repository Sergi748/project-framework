# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:48:12 2020

@author: sc250091
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class tratamiento():

    '''Clase para realizar el tratamiento del tablon'''
    def inputNA(self, df, path, nameProject, diccionario=""):
        
        df1 = df.copy()
        '''Se puede entregar un dataframe (['variable', 'valor']) con el valor de la imputacion para cada variable o no'''
        if diccionario.empty:
            dfSummaryImput = pd.DataFrame(columns=['Variable', 'Input', 'Type'])
            variables = df.columns.tolist()
            naNumeric = -99
            naCategoric = 'desconocido'
            for i in variables:
                if df1[i].dtype == 'object' and df1[i].isnull().sum() != 0:
                    df1[i] = df1[i].fillna(naCategoric)
                elif df1[i].dtype != 'object' and df1[i].isnull().sum() != 0:
                    df1[i] = df1[i].fillna(naNumeric)

            '''Creamos diccionario para cuando no sea dado'''
            for i in range(len(variables)):
                dfSummaryImput.loc[i,'Variable'] = variables[i]
                if df[variables[i]].dtype == 'object':        
                    dfSummaryImput.loc[i,'Input'] = naCategoric
                    dfSummaryImput.loc[i,'Type'] = 'Categorical'
                else:
                    dfSummaryImput.loc[i,'Input'] = naNumeric
                    dfSummaryImput.loc[i,'Type'] = 'Numerical'
            dfSummaryImput.to_csv(path + '/' + nameProject + '/governance/diccionarioInputNAs.csv', sep=';', index=False)
            
        else:
            dfDicc = diccionario
            dfDicc.to_csv(path + '/' + nameProject + '/governance/diccionarioInputNAs.csv', sep=';', index=False)
            for i in df1.columns:
                if i in dfDicc.Variable.tolist():
                    value = dfDicc.loc[dfDicc.Variable == i, 'Input'].item()
                    df1[i] = df1[i].fillna(value)                          
        return df1
    
    def createDummies(self, df, target, path, nameProject): 
        
        df1 = df.copy()
        dfSummaryDummies = pd.DataFrame(columns=['Variable', 'Dummies'])
        variables = df1.columns.tolist()
        for i in range(len(variables)):
            if (df1[variables[i]].dtype == 'object') & (variables[i] != target): 
                dfSummaryDummies.loc[i, 'Variable'] = variables[i]
                dfSummaryDummies.loc[i, 'Dummies'] = df1[variables[i]].unique().tolist() 
        
        dfSummaryDummies.reset_index(drop=True).to_csv(path + '/' + nameProject + '/governance/diccionarioDummies.csv', sep=';', index=False)

        for var in df1.columns:
            if (df1[var].dtype == 'object') & (var != target): 
                df1 = pd.get_dummies(df1, columns=[var], dtype=int)
        return df1
    
    def inputNAPred(self, df, path, nameProject):
        
        df1 = df.copy()
        dfDicc = pd.read_csv(path + '/' + nameProject + '/governance/diccionarioInputNAs.csv', sep=';')
        for i in df1.columns:
            if (i in dfDicc.Variable.tolist()) & (df1[i].isnull().sum() > 0):
                typeValue = dfDicc.loc[dfDicc.Variable == i, 'Type'].item()
                value = int(dfDicc.loc[dfDicc.Variable == i, 'Input'].item()) if typeValue == 'Numerical' else str(dfDicc.loc[dfDicc.Variable == i, 'Input'].item())
                df1[i] = df1[i].fillna(value)  
        return df1

    def createDummiesPred(self, df, target, Id):

        df1 = df.copy()
        for var in df1.columns:
            if (df1[var].dtype == 'object') & (var != target) & (var != Id): 
                df1 = pd.get_dummies(df1, columns=[var], dtype=int)
        
        return df1
