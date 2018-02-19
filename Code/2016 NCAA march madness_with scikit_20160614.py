from __future__ import print_function
import pandas as pd
import numpy as np
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def GrabData():
    
    game_to_train = pd.read_csv('game.to.train1.csv')  
    game_to_predict = pd.read_csv('game.to.predict.csv')     
    game_to_validation = pd.read_csv('game.to.validation1.csv')    
    return game_to_train,game_to_predict,game_to_validation

# Defining training and test sets
######################################
# Grab Data                          #
######################################
DataSets = GrabData()  
game_to_train,game_to_predict,game_to_validation = DataSets
game_to_train.fillna(method='pad',inplace=True)
game_to_predict.fillna(method='bfill',inplace=True)
game_to_validation.fillna(method='bfill',inplace=True)
Trainlabels = game_to_train['team1win'].values
Validationlabels = game_to_validation['team1win'].values
game_to_train.drop(['team1win','seasonteam1','seasonteam2'], inplace=True, axis=1)
game_to_validation.drop(['team1win','seasonteam1','seasonteam2'], inplace=True, axis=1)
game_to_predict.drop(['seasonteam1','seasonteam2'], inplace=True, axis=1)

#######################################
## Normalization                     #
#######################################

ss = StandardScaler()
game_to_train[game_to_train.columns] = np.round(ss.fit_transform(game_to_train), 4)
game_to_validation[game_to_validation.columns] = np.round(ss.fit_transform(game_to_validation), 4)
game_to_predict[game_to_predict.columns] = np.round(ss.transform(game_to_predict), 4)

#######################################
## Normalization                     #
#######################################

TrainSet_Data_x=game_to_train.values
ValidationSet_Data_x=game_to_validation.values
TrainSet_Data_y=Trainlabels
ValidationSet_Data_y=Validationlabels
TestSet_Data_x =game_to_predict.values

######################################
# Training                           #
######################################

print('Training...')
extc = LogisticRegression ()
extc.fit(TrainSet_Data_x,TrainSet_Data_y) 
x_pred = extc.predict_proba(TrainSet_Data_x)
print(x_pred)
print(log_loss(Trainlabels, np.clip(x_pred[:,1]*1.0088, 1e-6, 1-1e-6)))
print('Predict...')
y_pred = extc.predict_proba(TestSet_Data_x)
y_pred1 = extc.predict(TestSet_Data_x)
print(y_pred1)