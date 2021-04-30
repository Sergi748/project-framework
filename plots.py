# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:48:52 2020

@author: Sergio Campos
"""
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# %matplotlib inline
# import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, roc_auc_score, auc
import scikitplot as skplt


class createPlots():
    
    '''Clase para realizar diferentes plots''' 
    
    def plotFeaturesSelect(self, dfVars, path, nameProject, name):
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
        fig.savefig(path + '/' + nameProject + '/output/metricas/Features_selection_' + name + '.png')
        plt.close(fig)
        plt.show()
        
        
    def roc_auc_lift(self, X_test, y_test, model, target, name, path, nameProject):
    
        y_score = model.predict_proba(X_test)
        
        # Create ROC curve
        fpr, tpr, threshold = roc_curve(y_test, y_score[::,1])
        roc_auc = auc(fpr, tpr)
    
        plt.figure(figsize=(7,7))
        plt.title('ROC AUC Curve')
        plt.plot(fpr, tpr, 'b', label = 'AUC model= %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(path + '/' + nameProject + '/output/metricas/ROC_AUC_train_' + name + '.png')
        plt.close()
        plt.show()
        
        plt.figure(figsize=(10,10))
        skplt.metrics.plot_lift_curve(y_test, y_score)
        plt.savefig(path + '/' + nameProject + '/output/metricas/Lift_curve_train_' + name + '.png')
        plt.close()
        plt.show()    
        
    
    def roc_auc_lift_predict(self, dfPredReal, target, Id, path, nameProject, modelToPred, nameSavePred):
    
        dfPredScore = pd.read_csv(path + '/' + nameProject + '/output/predicciones/prediccion_model_' + modelToPred + '_' + nameSavePred + '.csv', sep=';')
        dfPred_real_score = pd.merge(dfPredReal, dfPredScore, on=Id, how='inner')
        dfPred_real_score['scores_to_0'] = 1 - dfPred_real_score.score
    
        fpr, tpr, threshold = roc_curve(dfPred_real_score[target], dfPred_real_score['score'])
        roc_auc = auc(fpr, tpr)
    
        plt.figure(figsize=(7,7))
        plt.title('ROC AUC Curve')
        plt.plot(fpr, tpr, 'b', label = 'AUC model= %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(path + '/' + nameProject + '/output/metricas/ROC_AUC_predict_' + modelToPred + '_' + nameSavePred + '.png')
        plt.close()
        plt.show()    
    
        plt.figure(figsize=(10,10))
        skplt.metrics.plot_lift_curve(dfPred_real_score[target], dfPred_real_score[['scores_to_0', 'score']].to_numpy())
        plt.savefig(path + '/' + nameProject + '/output/metricas/Lift_curve_predict_' + modelToPred + '_' + nameSavePred + '.png')
        plt.close()
        plt.show()
        
