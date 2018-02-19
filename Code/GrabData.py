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

game_to_train = pd.read_csv('game.to.train.csv')
team1_data_by_season = pd.read_csv('team1_data_by_season.csv')
team2_data_by_season = pd.read_csv('team2_data_by_season.csv')
gb1 = game_to_train.groupby(['season','team1'])
gb2 = team1_data_by_season.groupby(['team1_season','team1_team'])
groups1 = dict(list(gb1))
groups2 = dict(list(gb2))
game_to_train1 =pd.merge(game_to_train,team1_data_by_season,how='inner',on=['key'])
