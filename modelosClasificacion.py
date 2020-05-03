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
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
from plots import createPlots
import xgboost as xgb


class modelos():
    
    '''Clase para seleccion de variabless'''
    def __init__(self, df, target, test_size, path, nameModel, cv, score_metric):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.path = path
        self.nameModel =nameModel
        self.cv = cv 
        self.score_metric = score_metric
        # self.features_importance_LogReg = pd.DataFrame()
        # self.features_importance_RanFor = pd.DataFrame()
        # self.features_importance_xgb = pd.DataFrame()
        # self.features_importance_VotClass = pd.DataFrame()
        
    def __split(self):
        return self.df.loc[:, ~self.df.columns.isin([self.target])], self.df.loc[:, self.df.columns == self.target]
        
    def __trainTest(self):
        df_x, df_y = modelos.__split(self)
        # Guardamos las varibles usadas en el modelo
        pd.DataFrame(df_x.columns, columns=['Variables_used']).to_csv(self.path + '/' + self.nameModel + '/governance/varsTrain.csv', sep=';', index=False)
        # Dividimos el tablon entre las variables dependientes e independiente        
        return train_test_split(df_x, df_y, test_size=self.test_size, random_state=22)
    
    def __metricsROC(model, dfTr, yTr, dfTs, yTs, target, name, path, nameModel):
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
        
        createPlots().plotROC_AUC_train_test(dfTrain=dfTr, dfTest=dfTs, target=target, name=name, path=path, nameModel=nameModel)

    
    def __metricsLift_old(model, X, y, target, name, path, nameModel):
        X['score'] = model.predict_proba(X)[::,1]
        X[target] = y
        X = X[['score', target]]
        dfLift = createPlots().tableLift(df=X, score='score', target= target)
        # for i in [1, 0.1, 0.05]:
        #     createPlots().plotLift(dfLift, i, name, path=path, nameModel=nameModel)
        createPlots().plotLift(dfLift, name, path=path, nameModel=nameModel)


    def __metricsLift(model, dfTr, yTr, dfTs, yTs, target, name, path, nameModel):
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

        createPlots().plotLift(df=dfLiftTrain, name='Train_' + name, path=path, nameModel=nameModel)
        createPlots().plotLift(df=dfLiftTest, name='Test_' + name, path=path, nameModel=nameModel)     
    
    def modelLogisticRegression(self, parametrosLogReg):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
        
        model_lr = GridSearchCV(LogisticRegression().fit(X_train, y_train), param_grid=parametrosLogReg, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        features_importance_LogReg = pd.DataFrame(data={'Variables': X_train.columns, 'coef': abs(model_lr.best_estimator_.coef_[0])}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        features_importance_LogReg.to_csv(self.path + '/' + self.nameModel + '/output/modelos/features_selection_LogisticRegression.csv', sep=';', index=False)
        createPlots().plotFeaturesSelect(features_importance_LogReg, self.path, self.nameModel, 'LogisticRegression')
        pd.DataFrame(model_lr.cv_results_).to_csv(self.path + '/' + self.nameModel + '/output/modelos/gridSearch_LogisticRegression.csv', index=False, sep=';')                
        scoreTrain = model_lr.score(X_train, y_train)
        scoreTest = model_lr.score(X_test, y_test)
        filename = self.path + '/' + self.nameModel + '/output/modelos/model_LogisticRegression.pkl'
        pickle.dump(model_lr, open(filename, 'wb'))
        modelos.__metricsROC(model=model_lr, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LogisticRegression', path=self.path, nameModel=self.nameModel)
        modelos.__metricsLift(model=model_lr, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='LogisticRegression', path=self.path, nameModel=self.nameModel)
        return print('Scores modelo LogisticRegresion \ntrain: ' + str(scoreTrain) + '\ntest: ' + str(scoreTest))
                   
    def modelRandomForest(self, parametrosRandomForest):    
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)
        
        model_rf = GridSearchCV(RandomForestClassifier().fit(X_train, y_train), param_grid=parametrosRandomForest, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        features_importance_RanFor = pd.DataFrame(data={'Variables': X_train.columns, 'coef': model_rf.best_estimator_.feature_importances_}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        features_importance_RanFor.to_csv(self.path + '/' + self.nameModel + '/output/modelos/features_selection_RandomForest.csv', sep=';', index=False)
        createPlots().plotFeaturesSelect(features_importance_RanFor, self.path, self.nameModel, 'RandomForest')
        pd.DataFrame(model_rf.cv_results_).to_csv(self.path + '/' + self.nameModel + '/output/modelos/gridSearch_RandomForestClassifier.csv', index=False, sep=';')        
        scoreTrain = model_rf.score(X_train, y_train)
        scoreTest = model_rf.score(X_test, y_test)
        filename = self.path + '/' + self.nameModel + '/output/modelos/model_RandomForest.pkl'
        pickle.dump(model_rf, open(filename, 'wb'))
        modelos.__metricsROC(model=model_rf, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='RandomForest', path=self.path, nameModel=self.nameModel)
        modelos.__metricsLift(model=model_rf, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='RandomForest', path=self.path, nameModel=self.nameModel)
        return print('Scores modelo RandomForest \ntrain: ' + str(scoreTrain) + '\ntest: ' + str(scoreTest))          
    
    def modelGradientBoostingClassifier(self, parametrosGradBoosting):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_GradBoosting = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parametrosGradBoosting, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        features_importance_GradBoosting = pd.DataFrame(data={'Variables': X_train.columns, 'coef': model_GradBoosting.best_estimator_.feature_importances_}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        features_importance_GradBoosting.to_csv(self.path + '/' + self.nameModel + '/output/modelos/features_selection_GradBoosting.csv', sep=';', index=False)
        createPlots().plotFeaturesSelect(features_importance_GradBoosting, self.path, self.nameModel, 'GradBoosting')
        pd.DataFrame(model_GradBoosting.cv_results_).to_csv(self.path + '/' + self.nameModel + '/output/modelos/gridSearch_GradientBoostingClassifier.csv', index=False, sep=';')
        scoreTrain = model_GradBoosting.score(X_train, y_train)
        scoreTest = model_GradBoosting.score(X_test, y_test)
        filename = self.path + '/' + self.nameModel + '/output/modelos/model_GradientBoostingClassifier.pkl'
        pickle.dump(model_GradBoosting, open(filename, 'wb'))
        modelos.__metricsROC(model=model_GradBoosting, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='GradBoosting', path=self.path, nameModel=self.nameModel)
        modelos.__metricsLift(model=model_GradBoosting, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='GradBoosting', path=self.path, nameModel=self.nameModel)
        return print('Scores modelo GradientBoostingClassifier \ntrain: ' + str(scoreTrain) + '\ntest: ' + str(scoreTest))
       
    def modelVotingClassifier(self, estimatorsVotClass, parametrosvotClass):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_votClass = GridSearchCV(estimator=estimatorsVotClass, param_grid=parametrosvotClass, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        pd.DataFrame(model_votClass.cv_results_).to_csv(self.path + '/' + self.nameModel + '/output/modelos/gridSearch_VotingClassifier.csv', index=False, sep=';')
        scoreTrain = model_votClass.score(X_train, y_train)
        scoreTest = model_votClass.score(X_test, y_test)
        filename = self.path + '/' + self.nameModel + '/output/modelos/model_VotingClassifier.pkl'
        pickle.dump(model_votClass, open(filename, 'wb'))
        modelos.__metricsROC(model=model_votClass, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='VotingClassifier', path=self.path, nameModel=self.nameModel)
        modelos.__metricsLift(model=model_votClass, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='VotingClassifier', path=self.path, nameModel=self.nameModel)
        return print('Scores modelo VotingClassifier \ntrain: ' + str(scoreTrain) + '\ntest: ' + str(scoreTest))

    def modelXGBClassifier(self, parametrosxgb):
        X_train, X_test, y_train, y_test = modelos.__trainTest(self)

        model_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=parametrosxgb, cv=self.cv, scoring=self.score_metric).fit(X_train, y_train)
        features_importance_xgb = pd.DataFrame(data={'Variables': X_train.columns, 'coef': model_xgb.best_estimator_.feature_importances_}, columns=['Variables', 'coef']).sort_values('coef', ascending=False).reset_index(drop=True)
        features_importance_xgb.to_csv(self.path + '/' + self.nameModel + '/output/modelos/features_selection_XGBClassifier.csv', sep=';', index=False)
        createPlots().plotFeaturesSelect(features_importance_xgb, self.path, self.nameModel, 'XGBClassifier')
        pd.DataFrame(model_xgb.cv_results_).to_csv(self.path + '/' + self.nameModel + '/output/modelos/gridSearch_XGBClassifier.csv', index=False, sep=';')
        scoreTrain = model_xgb.score(X_train, y_train)
        scoreTest = model_xgb.score(X_test, y_test)
        filename = self.path + '/' + self.nameModel + '/output/modelos/model_XGBClassifier.pkl'
        pickle.dump(model_xgb, open(filename, 'wb'))
        modelos.__metricsROC(model=model_xgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='XGBClassifier', path=self.path, nameModel=self.nameModel)
        modelos.__metricsLift(model=model_xgb, dfTr=X_train, yTr=y_train, dfTs=X_test, yTs=y_test, target=self.target, name='XGBClassifier', path=self.path, nameModel=self.nameModel)
        return print('Scores modelo XGBClassifier \ntrain: ' + str(scoreTrain) + '\ntest: ' + str(scoreTest))
        
        
        
