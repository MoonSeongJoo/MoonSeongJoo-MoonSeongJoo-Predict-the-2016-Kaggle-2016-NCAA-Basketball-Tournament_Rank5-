from __future__ import print_function
import pandas as pd
import numpy as np
import csv

from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.optimizers import SGD, Adam, RMSprop 
from keras.utils import np_utils 
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2, activity_l2
import keras.callbacks

def GrabData():
    
    game_to_train = pd.read_csv('game.to.train2.csv')  
    game_to_predict = pd.read_csv('game.to.predict1.csv')     
    game_to_validation = pd.read_csv('game.to.validation2.csv')    
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
nb_classes = np.max(TrainSet_Data_y)+1
TrainSet_Data_y1 = np_utils.to_categorical(TrainSet_Data_y, nb_classes)
nb_classes = np.max(ValidationSet_Data_y)+1
ValidationSet_Data_y1 = np_utils.to_categorical(ValidationSet_Data_y, nb_classes)

model = Sequential()
model.add(Dense(10, input_shape=(76,),init='glorot_uniform')) 
#model.add(Dense(20, input_shape=(76,),W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01),init='glorot_uniform')) 
model.add(Activation('tanh')) 
model.add(Dropout(0.5)) 
model.add(Dense(15,init='glorot_uniform')) 
model.add(Activation('tanh')) 
#model.add(Dropout(0.5))
model.add(Dense(20)) 
model.add(Activation('tanh')) 
#model.add(Dropout(0.2))   
model.add(Dense(25)) 
model.add(Activation('tanh')) 
model.add(Dense(30)) 
model.add(Activation('tanh')) 
model.add(Dense(35)) 
model.add(Activation('tanh')) 
model.add(Dense(40)) 
model.add(Activation('tanh'))
model.add(Dense(45)) 
model.add(Activation('tanh'))
model.add(Dense(50)) 
model.add(Activation('tanh'))
model.add(Dense(55)) 
model.add(Activation('tanh'))
model.add(Dense(60)) 
model.add(Activation('tanh'))
model.add(Dense(65)) 
model.add(Activation('tanh'))
model.add(Dense(70)) 
model.add(Activation('tanh'))
model.add(Dense(2)) 
model.add(Activation('softmax')) 

 
model.summary() 

sgd = SGD(lr=0.7, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='binary_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy']) 

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto') 

#callbacks = [
##    EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
#    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
#    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
#]
history = model.fit(TrainSet_Data_x, TrainSet_Data_y1, 
                    batch_size=500, nb_epoch=285, 
                    verbose=2,validation_data=(ValidationSet_Data_x,ValidationSet_Data_y1),callbacks=[earlyStopping])

model.save_weights('my_model_weights.h5')
#score = model.evaluate(ValidationSet_Data_x,ValidationSet_Data_y1,
#                       batch_size=100)

preds = model.predict(TestSet_Data_x)
preds1 = model.predict_classes(TestSet_Data_x)
print('Printing TestSet_Data_x...')
print(preds[:50])
print(preds1[:50])

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.Pred = preds[:,1]
sample_submission.to_csv('submission.csv', index=False)
