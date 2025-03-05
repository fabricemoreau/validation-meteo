import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import joblib

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
nb_jours = 12 # il faut qu'il soit divisible par 4
seed = 42

periode = '2010-2024'
path    = './data/processed/autoenc'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')
# on ne fait pas de : cela sera géré dans la construction du jeu : meteobydate = meteobydate.dropna()
meteobydate = meteobydate.sort_values(['codearvalis', 'datemesure'])


anomalies_dict['TN_anomalie_pred']