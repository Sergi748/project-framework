# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:48:52 2020

@author: sc250091
"""
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class createPlots():

    '''Clase para realizar diferentes plots'''
    # Funcion creacion tablon para grafica lift
    def tableLift(self, df, score, target):
        df = df.loc[:,[score, target]]
        df = df.sample(df.shape[0], random_state=22)
        df[target] = df[target].astype(int)
        #generar variables y calculos para pintar los resultados de lift
        df = df.sort_values(score, ascending=False).reset_index()
        df['cum_sum'] = df[target].cumsum()
        df['poblacion_tpu'] = (df.index + 1)/df.shape[0]
        df['aleatorio'] = df['poblacion_tpu']*df.cum_sum.max()
        df['precision'] = df['cum_sum'] / (df.index + 1)
        df['lift'] = df['cum_sum']/df['aleatorio']
        return df

    # Funcion para plot lift
    def plotLift_old(self, df, pct, name, path, nameModel):
        corte_izq = 1
        corte_derecho = int(df.shape[0]*pct)
        fig = plt.figure(figsize=(8, 7))
        plt.title('Curva lift con el ' + str(int(pct * 100)) + ' % de la poblacion')
        plt.scatter(df['poblacion_tpu'][(corte_izq - 1):(corte_derecho + 1)], df['lift'][(corte_izq - 1):(corte_derecho + 1)])
        plt.xlabel('poblacion')
        plt.ylabel('lift')
        plt.savefig(path + '/' + nameModel + '/output/metricas/Lift_' + name + '_' + str(int(pct * 100)) + '.png')
        plt.close(fig)
        plt.show()
            
    # Funcion para plot lift
    def plotLift(self, df, name, path, nameModel):
        corte_izq = 1
        corte_derecho = int(df.shape[0]*1)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.scatter(df['poblacion_tpu'][(corte_izq - 1):(corte_derecho + 1)], df['lift'][(corte_izq - 1):(corte_derecho + 1)])
        ax1.title.set_text('Lift 100 % poblacion')
        corte_derecho = int(df.shape[0]*0.1)
        ax2.scatter(df['poblacion_tpu'][(corte_izq - 1):(corte_derecho + 1)], df['lift'][(corte_izq - 1):(corte_derecho + 1)])
        ax2.title.set_text('Lift 10 % poblacion')
        plt.savefig(path + '/' + nameModel + '/output/metricas/Lift_' + name + '.png')
        plt.close(fig)
        plt.show()
    
    
    def plotROC_AUC_old(self, df, target, name, path, nameModel):
        fpr24ghz, tpr24ghz, thresholds = roc_curve(df[target], df['score'])
        fig = plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1])
        plt.plot(fpr24ghz, tpr24ghz);
        plt.title('AUC-ROC')
        plt.xlabel('FPR', color='#1C2833')
        plt.ylabel('TPR', color='#1C2833')
        plt.grid()
        plt.text(0.61, 0.01, str('AUC-ROC = ') + str(round(100*(roc_auc_score(df[target], df['score'])), 3)) + str('%'))
        plt.savefig(path + '/' + nameModel + '/output/metricas/ROC_AUC_' + name + '.png')
        plt.close(fig)
        plt.show()
        
    def plotROC_AUC_train_test(self, dfTrain, dfTest, target, name, path, nameModel):
        fpr24ghz, tpr24ghz, thresholds = roc_curve(dfTrain[target], dfTrain['score'])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot([0, 1], [0, 1])
        ax1.plot(fpr24ghz, tpr24ghz)
        ax1.title.set_text('AUC-ROC train')
        ax1.grid()
        ax1.text(0.3, 0.07, str('AUC-ROC = ') + str(round(100*(roc_auc_score(dfTrain[target], dfTrain['score'])), 3)) + str('%'), transform=ax1.transAxes)
        fpr24ghz, tpr24ghz, thresholds = roc_curve(dfTest[target], dfTest['score'])
        ax2.plot([0, 1], [0, 1])
        ax2.plot(fpr24ghz, tpr24ghz)
        ax2.title.set_text('AUC-ROC test')
        ax2.grid()
        ax2.text(0.3, 0.07, str('AUC-ROC = ') + str(round(100*(roc_auc_score(dfTest[target], dfTest['score'])), 3)) + str('%'), transform=ax2.transAxes)
        fig.text(0.5, 0.03, 'FPR', ha='center')
        fig.text(0.05, 0.5, 'TPR', va='center', rotation='vertical')     
        plt.savefig(path + '/' + nameModel + '/output/metricas/ROC_AUC_train_test_' + name + '.png')
        plt.close(fig)
        plt.show()
        
    def plotFeaturesSelect(self, dfVars, path, nameModel, name):
        plt.rcdefaults()
        fig, ax = plt.subplots()
        
        y_axis = dfVars.Variables
        y_pos = np.arange(dfVars.shape[0])
        values = dfVars.coef
        
        ax.barh(y_pos, values, align='center', color='green')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_axis)
        ax.invert_yaxis()
        ax.set_xlabel('Values')
        ax.set_ylabel('FEATURES')
        ax.set_title('Features importances')    
        fig.savefig(path + '/' + nameModel + '/output/metricas/Features_selection_' + name + '.png')
        plt.close(fig)
        plt.show()
