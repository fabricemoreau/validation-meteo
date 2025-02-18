import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

periode = '2010-2012'
fichier = './data/raw/donneesmeteo_' + periode + '_completes.csv'
meteodf = pd.read_csv(fichier, sep = ';')

print("pas de valeur manquante excepté valeurorigine et idmodeobtention")
print(meteodf.isna().sum())

# une valeur d'origine indiquée implique que la donnée a été corrigée
print("Statut des données qui ont une valeurorigine renseignée:", meteodf[meteodf.valeurorigine.notna()].idstatut.value_counts())
# des données ont été corrigées sans valeur d'origine (données manquantes)
print("Pourcentage de valeur d'origine manquante quand la donnée a été corrigée:", meteodf[meteodf.idstatut > 1].valeurorigine.isna().sum() / meteodf[meteodf.idstatut > 1].shape[0])

# très peu de valeur d'origine pour TM
meteodf[meteodf.libellecourt == 'TM'].valeurorigine.count()

# on peut donc supprimer TM
meteodf = meteodf[meteodf.libellecourt != 'TM']

# on considère les -999 des valeurs d'origine comme une absence de données: on ne peut pas exploiter ces lignes
# on les conserve pour la dataviz
meteodf.loc[meteodf.valeurorigine == -999, 'valeurorigine'] = np.nan

# transformation des dates
meteodf['datemesure'] = pd.to_datetime(meteodf.datemesure)

# Il ne faut retenir qu'une valeur par station jour et paramètre. On retient en priorité idmodeobtention le plus petit
meteodf.fillna({'idmodeobtention': 4}, inplace = True) # on prend les na en dernier recours
meteodf = meteodf.sort_values(by = ['datemesure', 'codearvalis', 'libellecourt', 'idmodeobtention'])
meteodf.drop_duplicates(subset =['datemesure', 'codearvalis', 'libellecourt'], keep = 'first', inplace = True)

#idstatut et idmodeobtention
meteodf.reset_index(drop = True, inplace = True)
meteodf =meteodf.drop(columns = ['idstatut', 'idmodeobtention'])

# introduction de la variable à prédire: correction
meteodf['correction'] = np.abs(meteodf.valeur - meteodf.valeurorigine) > 0
meteodf['correction'] = np.where(meteodf['correction'], 1, 0)
meteodf['correction'] = meteodf['correction'].fillna(0)

print('plus de duplication : ',  meteodf.duplicated(['codearvalis', 'datemesure', 'libellecourt']).sum())

# on pivote pour avoir une ligne par date
meteobydate = meteodf.pivot(index = ['codearvalis', 'datemesure'], columns = 'libellecourt')
meteobydate.to_csv('./data/processed/meteo_2010-2012.csv', sep=';')
meteobydate = pd.read_csv('./data/processed/meteo_2010-2012.csv', sep=';', header = [0,1, 2])
meteobydate.reset_index(inplace = True)
# pas de données manquantes
meteobydate.valeur.TN.isna().sum()
##################################### étude paramètres

def graph_repartition(param_meteo, nom_param_meteo, libelle_echelle_y):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.boxplot(y = param_meteo, data = meteobydate.valeur, ax = axes[0])
    axes[0].set_title("Variation de " + nom_param_meteo + ", période " + periode)
    axes[0].set_ylabel(libelle_echelle_y)
    #axes[0].xlabel("Stations")

    sns.histplot(x = param_meteo, data = meteobydate.valeur, ax = axes[1])
    axes[1].set_title("distribution de " + nom_param_meteo + ", période " + periode)
    axes[1].set_ylabel(libelle_echelle_y)
    #axes[1].xlabel("Altitude (m)")
    plt.savefig('./reports/figures/' + param_meteo + '_' + periode + '.png')
    plt.show();
graph_repartition('TN', 'Température minimale (TN)', 'Température (°C)')
graph_repartition('TX', 'Température maximale (TN)', 'Température (°C)')
graph_repartition('RR', 'Précipitations (RR)', 'Pluie (mm)')
graph_repartition('GLOT', 'Rayonnement global (GLOT)', 'Rayonnement (J/cm²)')
graph_repartition('ETP', 'Evapotranspiration potentielle (ETP)', 'mm')

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'TN', y = 'TX', data = meteobydate.valeur, label = "TN en fonction de TX")
sns.lineplot(x = [-30, 40], y = [-30, 40], label = "première bissectrice")
plt.title("TN en fonction de TX, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/TN_fonction_TX_' + periode + '.png')
plt.show();

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'TN', y = 'GLOT', data = meteobydate.valeur, label = "TN en fonction de GLOT")
plt.title("TN en fonction de GLOT, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/TN_fonction_GLOT_' + periode + '.png')
plt.show();

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'ETP', y = 'GLOT', data = meteobydate.valeur, label = "ETP en fonction de GLOT")
plt.title("ETP en fonction de GLOT, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/ETP_fonction_GLOT_' + periode + '.png')
plt.show();


meteobydate.valeur.corr(method = "spearman")
from scipy.stats import pearsonr
pearsonr(meteobydate.valeur.TN, meteobydate.valeur.TX) # corrélation
pearsonr(meteobydate.valeur.TN.tolist(), meteobydate.valeur.ETP.tolist()) # corrélation
