import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

path    = './data/processed/autoencsimple'

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
parametres_origine = [ param + '_origine' for param in parametres ]
parametres_anomalie = [ param + '_anomalie' for param in parametres ]
parametres_anomalie_pred = [ param + '_anomalie_pred' for param in parametres ]

valid_df = pd.read_csv(path + '/meteobydate_valid_pred.csv', sep = ';')
valid_df['anomalie_pred'] = valid_df[parametres_anomalie].sum(axis = 1)
valid_df['anomalie_pred'] = np.where(valid_df['anomalie_pred'] > 0, 1, 0)
valid_df['anomalie'] = np.where(valid_df['anomalie'] > 0, 1, 0)

print(pd.crosstab(valid_df['anomalie'], valid_df['anomalie_pred']))
print(classification_report(valid_df['anomalie'], valid_df['anomalie_pred']))
# l'autoencodeur est bon pour prédire les jours où il y a une anomalie, mais pas forcément sur le bon paramètre
# (si l'ETP est anormal, il peut prédire un ETP normal, mais générer une TX anormale)

# Etude des FN
fn_df = valid_df[(valid_df['anomalie'] == 1) & (valid_df['anomalie_pred'] == 0)]
plt.figure(figsize = (20, 20))
for param in parametres:
    ax = plt.subplot(2, 2, parametres.index(param) + 1)
    ax.set_title(param)
    ax.plot(fn_df[param], fn_df[param + '_pred'], 'o')
plt.savefig(path + '/analyse_fn.png')
plt.show();
# on pourrait d'abord identifier les jours que l'autoencodeur considère comme anomalie, puis regarder les paramètres
anomalies_df_TX = valid_df[valid_df.TX_anomalie == True]
print(pd.crosstab(anomalies_df_TX['anomalie'], anomalies_df_TX['anomalie_pred']))
