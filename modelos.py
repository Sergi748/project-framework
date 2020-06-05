# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:50:11 2020

@author: sc250091
"""

import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
from plots import createPlots
from metrics import metrics
import xgboost as xgb
import lightgbm as lgb
from metrics import *


class modelos():
    
    '''Clase de modelos'''
    def __init__(self, df, target, test_size, path, nameProject, cv, score_metric):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.path = path
        self.nameProject = nameProject
        self.cv = cv 
        self.score_metric = score_metric
        
        
    def __split(self):
        return self.df.loc[:, ~self.df.columns.isin([self.target])], self.df.loc[:, self.df.columns == self.target]
        
    
    def __trainTest(self):
        df_x, df_y = modelos.__split(self)
        # Dividimos el tablon entre las variables dependientes e independiente        
        return train_test_split(df_x, df_y, test_size=self.test_size, random_state=22)
    
    
    def __metricsROC(model, dfTr, yTr, dfTs, yTs, target, name, path, nameProject):
        dfTr=dfTr.copy()
        dfTs=dfTs.copy()

        # Train
        dfTr['score'] = model.predict_proba(dfTr)[::,1]
        dfTr[target] = yTr
        dfTr = dfTr[['score', target]]
        dfTr[target] = dfTr[target].astype(int)
        # Test
        dfTs['score'] = model.predict_proba(dfTs)[::,1]
        dfTs[target] = yTs
        dfTs = dfTs[['score', target]]
        dfTs[target] = dfTs[target].astype(int)
        
        createPlots().plotROC_AUC_train_test(dfTrain=dfTr, dfTest=dfTs, target=target, name=name, path=path, nameProject=nameProject)

    
    def __metricsLift_old(model, X, y, target, name, path, nameProject):
        X['score'] = model.predict_proba(X)[::,1]
        X[target] = y
        X = X[['score', target]]
        dfLift = createPlots().tableLift(df=X, score='score', target= target)
        # for i in [1, 0.1, 0.05]:
        #     createPlots().plotLift(dfLift, i, name, path=path, nameProject=nameProject)
        createPlots().plotLift(dfLift, name, path=path, nameProject=nameProject)


    def __metricsLift(model, dfTr, yTr, dfTs, yTs, target, name, path, nameProject):
        dfTr=dfTr.copy()
        dfTs=dfTs.copy()

        # Train
        dfTr['score'] = model.predict_proba(dfTr)[::,1]
        dfTr[target] = yTr
        dfTr = dfTr[['score', target]]
        # Test
        dfTs['score'] = model.predict_proba(dfTs)[::,1]
        dfTs[target] = yTs
        dfTs = dfTs[['score', target]]
        
        dfLiftTrain = createPlots().tableLift(df=dfTr, score='score', target= target)
        dfLiftTest = createPlots().tableLift(df=dfTs, score='score', target= target)

        createPlots().plotLift(df=dfLiftTrain, name='Train_' + name, path=path, nameProject=nameProject)
        createPlots().plotLift(df=dfLiftTest, name='Test_' + name, path=path, nameProject=nameProject)     
    
    
    def __featuresAndPickle(model, target, X, path, nameProject, modelName):
        if modelName == 'LogisticRegression':
            features_importance = pd.DataFrame(data={'Variables': X.columns, 'coef': abs(model.best_estimator_.coef_[0])}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        else:
            features_importance = pd.DataFrame(data={'Variables': X.columns, 'coef': model.best_estimator_.feature_importances_}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        features_importance.to_csv(path + '/' + nameProject + '/output/modelos/features_selection_' + modelName + '.csv', sep=';', index=False)
        createPlots().plotFeaturesSelect(features_importance, path, nameProject, modelName)
        pd.DataFrame(model.cv_results_).to_csv(path + '/' + nameProject + '/output/modelos/gridSearch_' + modelName + '.csv', index=False, sep=';')                
        filename = path + '/' + nameProject + '/output/modelos/model_' + modelName + '.pkl'
        pickle.dump(model, open(filename, 'wb'))
   
        
    def modelLogisticRegression(self, parametrosLogReg):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
        
        model_lr = GridSearchCV(LogisticRegression(), param_grid=parametrosLogReg, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__featuresAndPickle(model=model_lr, target=self.target, X=X_train, path=self.path, nameProject=self.nameProject, modelName='LogisticRegression')       
        modelos.__metricsROC(model=model_lr, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LogisticRegression', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_lr, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LogisticRegression', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_lr, path=self.path, nameProject=self.nameProject, name='LogisticRegression')
        return print('Scores modelo LogisticRegresion \ntrain: ' + str(model_lr.score(X_train, y_train)) + '\ntest: ' + str(model_lr.score(X_test, y_test)))
                   
    
    def modelRandomForest(self, parametrosRandomForest):    
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
        
        model_rf = GridSearchCV(RandomForestClassifier(), param_grid=parametrosRandomForest, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__featuresAndPickle(model=model_rf, target=self.target, X=X_train, path=self.path, nameProject=self.nameProject, modelName='RandomForest')       
        modelos.__metricsROC(model=model_rf, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='RandomForest', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_rf, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='RandomForest', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_rf, path=self.path, nameProject=self.nameProject, name='RandomForest')
        return print('Scores modelo RandomForest \ntrain: ' + str(model_rf.score(X_train, y_train)) + '\ntest: ' + str(model_rf.score(X_test, y_test)))          
    
    
    def modelGradientBoostingClassifier(self, parametrosGradBoosting):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_GradBoosting = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parametrosGradBoosting, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__featuresAndPickle(model=model_GradBoosting, target=self.target, X=X_train, path=self.path, nameProject=self.nameProject, modelName='GradientBoostingClassifier')       
        modelos.__metricsROC(model=model_GradBoosting, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='GradBoosting', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_GradBoosting, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='GradBoosting', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_GradBoosting, path=self.path, nameProject=self.nameProject, name='GradBoosting')
        return print('Scores modelo GradientBoostingClassifier \ntrain: ' + str(model_GradBoosting.score(X_train, y_train)) + '\ntest: ' + str(model_GradBoosting.score(X_test, y_test)))
       
        
    def modelVotingClassifier(self, estimatorsVotClass, parametrosvotClass):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_votClass = GridSearchCV(estimator=estimatorsVotClass, param_grid=parametrosvotClass, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(model_votClass.cv_results_).to_csv(self.path + '/' + self.nameProject + '/output/modelos/gridSearch_VotingClassifier.csv', index=False, sep=';')
        filename = self.path + '/' + self.nameProject + '/output/modelos/model_VotingClassifier.pkl'
        pickle.dump(model_votClass, open(filename, 'wb'))
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__metricsROC(model=model_votClass, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='VotingClassifier', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_votClass, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='VotingClassifier', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_votClass, path=self.path, nameProject=self.nameProject, name='VotingClassifier')
        return print('Scores modelo VotingClassifier \ntrain: ' + str(model_votClass.score(X_train, y_train)) + '\ntest: ' + str(model_votClass.score(X_test, y_test)))


    def modelXGBClassifier(self, parametrosxgb):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=parametrosxgb, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__featuresAndPickle(model=model_xgb, target=self.target, X=X_train, path=self.path, nameProject=self.nameProject, modelName='XGBClassifier')       
        modelos.__metricsROC(model=model_xgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='XGBClassifier', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_xgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='XGBClassifier', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_xgb, path=self.path, nameProject=self.nameProject, name='XGBClassifier')
        return print('Scores modelo XGBClassifier \ntrain: ' + str(model_xgb.score(X_train, y_train)) + '\ntest: ' + str(model_xgb.score(X_test, y_test)))        
        
    
    def modelLightGBMClassifier(self, parametroslgb):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
    
        model_lgb = GridSearchCV(estimator=lgb.LGBMClassifier(), param_grid=parametroslgb, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__featuresAndPickle(model=model_lgb, target=self.target, X=X_train, path=self.path, nameProject=self.nameProject, modelName='LGBMClassifier')       
        modelos.__metricsROC(model=model_lgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LGBMClassifier', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_lgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LGBMClassifier', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_lgb, path=self.path, nameProject=self.nameProject, name='LGBMClassifier')
        return print('Scores modelo LGBMClassifier \ntrain: ' + str(model_lgb.score(X_train, y_train)) + '\ntest: ' + str(model_lgb.score(X_test, y_test)))
     
        
    def modelKNeighborsClassifier(self, parametrosknn):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
    
        model_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametrosknn, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(model_knn.cv_results_).to_csv(self.path + '/' + self.nameProject + '/output/modelos/gridSearch_KnnClassifier.csv', index=False, sep=';')
        filename = self.path + '/' + self.nameProject + '/output/modelos/model_KnnClassifier.pkl'
        pickle.dump(model_knn, open(filename, 'wb'))
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        modelos.__metricsROC(model=model_knn, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='KnnClassifier', path=self.path, nameProject=self.nameProject)
        modelos.__metricsLift(model=model_knn, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='KnnClassifier', path=self.path, nameProject=self.nameProject)
        metrics().metricsClassifier(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_knn, path=self.path, nameProject=self.nameProject, name='KnnClassifier')
        return print('Scores modelo KnnClassifier \ntrain: ' + str(model_knn.score(X_train, y_train)) + '\ntest: ' + str(model_knn.score(X_test, y_test)))
    
    
    def modelLinearRegression(self, parametroslinReg):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
            
        model_linReg = GridSearchCV(LinearRegression(), param_grid=parametroslinReg, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(model_linReg.cv_results_).to_csv(self.path + '/' + self.nameProject + '/output/modelos/gridSearch_LinearRegression.csv', index=False, sep=';')
        filename = self.path + '/' + self.nameProject + '/output/modelos/model_LinearRegression.pkl'
        pickle.dump(model_linReg, open(filename, 'wb'))
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        metrics().metricsRegression(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_linReg, path=self.path, nameProject=self.nameProject, name='LinearRegression')
        return print('Scores modelo LinearRegression \ntrain: ' + str(model_linReg.score(X_train, y_train)) + '\ntest: ' + str(model_linReg.score(X_test, y_test)))

        
    def modelXGBRegressor(self, parametrosxgbReg):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
    
        model_xgbReg = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=parametrosxgbReg, cv=5, scoring='r2').fit(X_train, y_train)
        pd.DataFrame(model_xgbReg.cv_results_).to_csv(self.path + '/' + self.nameProject + '/output/modelos/gridSearch_XGBRegression.csv', index=False, sep=';')
        filename = self.path + '/' + self.nameProject + '/output/modelos/model_XGBRegression.pkl'
        pickle.dump(model_xgbReg, open(filename, 'wb'))
        pd.DataFrame(X_train.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameProject + '/governance/varsTrain.csv', sep=';', index=False)
        metrics().metricsRegression(dfXTr=X_train, dfYTr=y_train, dfXTs=X_test, dfYTs=y_test, target=self.target, model=model_xgbReg, path=self.path, nameProject=self.nameProject, name='XGBRegression')
        return print('Scores modelo XGBRegressor \ntrain: ' + str(model_xgbReg.score(X_train, y_train)) + '\ntest: ' + str(model_xgbReg.score(X_test, y_test)))    

