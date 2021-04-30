# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:45:21 2020

@author: Sergio Campos
"""

# Librerias
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score



class metrics():
    
    def __createdfMetricsClassifier(self, df, target):
        dfMetrics = pd.DataFrame(columns=['metrica', 'valor'])
        dfMetrics.metrica = ['Accuracy score', 'F1 score', 'Log loss'
                             , 'Precision score', 'Recall score', 'ROC AUC score']
        dfMetrics.valor = [accuracy_score(df[target], df.score.round()), f1_score(df[target], df.score.round())
                           , log_loss(df[target], df.score.round()), precision_score(df[target], df.score.round())
                           , recall_score(df[target], df.score.round()), roc_auc_score(df[target], df.score)]
        
        return dfMetrics


    def __saveExcelTrain(self, dfTr, dfTs, path, nameProject, name):
        # Guardado metricas en fichero Excel
        writer = pd.ExcelWriter(path + '/' + nameProject + '/output/metricas/metrics_' + str(name) + '.xlsx', engine='xlsxwriter')
        dfTr.to_excel(writer, sheet_name='Train', index=False)
        dfTs.to_excel(writer, sheet_name='Test', index=False)
        writer.save()

    
    def metricsRegression(self, dfXTr, dfYTr, dfXTs, dfYTs, target, model, path, nameProject, name):
        
        def __createdf(dfX, dfY, target, model):    
            df1 = dfX.copy()
            y = dfY.copy()
            
            df1['score'] = model.predict(df1)
            df1[target] = y
            df1 = df1[['score', target]]
            df1[target] = df1[target].astype(int)
            
            return df1
        
        def __createdfMetrics(df, target):
            dfMetrics = pd.DataFrame(columns=['metrica', 'valor'])
            dfMetrics.metrica = ['Mean absolute error', 'Mean squared error', 'R2'
                                 , 'Explained variance', 'Max error', 'Mean squared log error'
                                 , 'Median absolute error', 'Mean poisson deviance'
                                 , 'Mean gamma deviance', 'Mean tweedie deviance']
            dfMetrics.valor = [mean_absolute_error(df[target], df.score)
                               , mean_squared_error(df[target], df.score)
                               , r2_score(df[target], df.score)
                               , explained_variance_score(df[target], df.score)
                               , max_error(df[target], df.score)
                               , mean_squared_log_error(df[target], df.score)
                               , median_absolute_error(df[target], df.score)
                               , mean_poisson_deviance(df[target], df.score)
                               , mean_gamma_deviance(df[target], df.score)
                               , mean_tweedie_deviance(df[target], df.score)]
            
            return dfMetrics
            
        dfTr = __createdf(dfXTr, dfYTr, target, model)    
        dfTs = __createdf(dfXTs, dfYTs, target, model)    
        dfMetricsTr = __createdfMetrics(dfTr, target)  
        dfMetricsTs = __createdfMetrics(dfTs, target)  

        metrics().__saveExcelTrain(dfMetricsTr, dfMetricsTs, path, nameProject, name)


    def metricsClassifier(self, dfXTr, dfYTr, dfXTs, dfYTs, target, model, path, nameProject, name):
            
        def __createdf(dfX, dfY, target, model):    
            df1 = dfX.copy()
            y = dfY.copy()
            
            df1['score'] = model.predict_proba(df1)[::,1]
            df1[target] = y
            df1 = df1[['score', target]]
            df1[target] = df1[target].astype(int)
            
            return df1
                    
        dfTr = __createdf(dfXTr, dfYTr, target, model)    
        dfTs = __createdf(dfXTs, dfYTs, target, model)    
        dfMetricsTr = metrics().__createdfMetricsClassifier(dfTr, target)  
        dfMetricsTs = metrics().__createdfMetricsClassifier(dfTs, target)  
    
        metrics().__saveExcelTrain(dfMetricsTr, dfMetricsTs, path, nameProject, name)

    
    def metricsClassifierPred(self, df, target, path, nameProject, name):
                                
        dfMetrics = metrics().__createdfMetricsClassifier(df, target)  
        # Guardado metricas en fichero Excel
        writer = pd.ExcelWriter(path + '/' + nameProject + '/output/metricas/metrics_' + str(name) + '.xlsx', engine='xlsxwriter')
        dfMetrics.to_excel(writer, sheet_name='Pred_' + name, index=False)
        writer.save()
    
    

