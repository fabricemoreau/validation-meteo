''' Clustering: DBSCAN'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

periode = '2010-2024'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']



col_to_keep = [s + '_origine' for s in parametres]
#for i in range(1, 6):
#    col_to_keep.extend([s + '_' + str(i) for s in parametres])
col_to_keep.extend(['Altitude', 'Lambert93x', 'Lambert93y', 'datemesure'])
df = meteobydate[col_to_keep]
y = (meteobydate.anomalie > 1).astype(int)

del meteobydate

# on regarde un paramètre en particulier
X = df.iloc[:(len(df) // 5)]
y = y[:(len(df) // 5)]

X = X.astype({'datemesure': 'int64'})

sc = StandardScaler()
X_norm =sc.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns = X.columns)

model = DBSCAN(eps = 0.4, min_samples = 15, n_jobs = -1)
model.fit(X_norm.values)

param_model = { 'eps': [0.4, 0.8, 1.2, 1.8, 2],
                'min_samples' : [100, 500, 1000],
                'n_jobs': [-1]
                }

param_grid = ParameterGrid(param_model)
accuracy = np.empty(len(param_grid))
roc = np.empty(len(param_grid))
mcc = np.empty(len(param_grid))

for i in range(0, len(param_grid)):
    print(param_grid[i])
    model = DBSCAN(**(param_grid[i]))
    model.fit(X_norm.values)
    y_pred = model.labels_ # algorithme de clustering
    y_pred[y_pred == -1] = 1
    y_pred[y_pred!=-1] = 0
    accuracy[i] = accuracy_score(y, y_pred)
    roc[i] =  roc_auc_score(y, y_pred)
    mcc[i] = matthews_corrcoef(y, y_pred)
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
    print(pd.crosstab(y, y_pred))
    print('roc', roc_auc_score(y, y_pred))
    print('MCC Score', matthews_corrcoef(y, y_pred))
