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


##################################### étude paramètres
periode = '2010-2024'
fichier = './data/processed/meteo_pivot_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';')

def graph_repartition(df, param_meteo, nom_param_meteo, libelle_echelle_y):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.boxplot(y = param_meteo, data = df, ax = axes[0])
    axes[0].set_title("Variation de " + nom_param_meteo + ", période " + periode)
    axes[0].set_ylabel(libelle_echelle_y)
    #axes[0].xlabel("Stations")

    sns.histplot(x = param_meteo, data = df, ax = axes[1])
    axes[1].set_title("distribution de " + nom_param_meteo + ", période " + periode)
    axes[1].set_xlabel(libelle_echelle_y)
    axes[1].set_ylabel("Effectif")
    #axes[1].xlabel("Altitude (m)")
    plt.savefig('./reports/figures/' + param_meteo + '_' + periode + '.png')
    plt.show();
graph_repartition(meteobydate, 'TN', 'Température minimale (TN)', 'Température (°C)')
graph_repartition(meteobydate, 'TX', 'Température maximale (TN)', 'Température (°C)')
graph_repartition(meteobydate[meteobydate.RR > 0], 'RR', 'Précipitations (RR) positives', 'Pluie (mm)')
graph_repartition(meteobydate[(meteobydate.RR < 5.6) & meteobydate.RR > 0], 'RR', 'Précipitations (RR) non extrêmes', 'Pluie (mm)')
graph_repartition(meteobydate, 'GLOT', 'Rayonnement global (GLOT)', 'Rayonnement (J/cm²)')
graph_repartition(meteobydate, 'ETP', 'Evapotranspiration potentielle (ETP)', 'mm')

# Etude des valeurs extrêmes:
# ETP
print(meteobydate[meteobydate.ETP > 8].TX.describe())
# ETP extrême les jours où les autres paramètres favorisant l'ETP ne sont pas extrêmes
print(meteobydate[(meteobydate.ETP > 8) & (meteobydate.TX < 25) & (meteobydate.RR < 15) & (meteobydate.GLOT < 3000)])
# ETP avec température à 16.7 suspect
# pluies
plt.figure(figsize = (10,5))
meteobydate['presence_RR'] = (meteobydate.RR > 0)
sns.countplot(x = "presence_RR", data = meteobydate)
plt.savefig('./reports/figures/RR_presence_' + periode + '.png')
plt.show();
meteobyrainday = meteobydate[meteobydate.RR > 0]
print('cumul de pluie 75% des jours pluvieux: ', meteobyrainday.RR.describe()['75%'])
plt.figure(figsize = (5,5))
g = sns.histplot(x = 'RR', data = meteobyrainday[meteobyrainday.RR > 5.6], bins = [5, 10, 15, 20, 30, 50])
plt.set_title("distribution de pluies extrêmes " + periode  + "periode")
plt.set_xlabel(libelle_echelle_y)
plt.set_ylabel("Effectif")
g.set_xticks([5, 10, 15, 20, 30, 50], labels = ['5', '10', '15', '20', '30', '500'])
plt.savefig('./reports/figures/RR_extreme_' + periode + '.png')
plt.show();

###########################

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'TN', y = 'TX', data = meteobydate, label = "TN en fonction de TX")
sns.lineplot(x = [-30, 40], y = [-30, 40], label = "première bissectrice")
plt.title("TN en fonction de TX, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/TN_fonction_TX_' + periode + '.png')
plt.show();

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'TX', y = 'GLOT', data = meteobydate, label = "TX en fonction de GLOT")
plt.title("TN en fonction de GLOT, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/TX_fonction_GLOT_' + periode + '.png')
plt.show();

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'ETP', y = 'GLOT', data = meteobydate, label = "ETP en fonction de GLOT")
plt.title("ETP en fonction de GLOT, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/ETP_fonction_GLOT_' + periode + '.png')
plt.show();

plt.figure(figsize=(5, 5))
sns.scatterplot(x = 'TX', y = 'RR', data = meteobydate, label = "RR en fonction de TX")
plt.title("RR en fonction de TX, periode " + periode)
plt.legend()
plt.savefig('./reports/figures/RR_fonction_TX_' + periode + '.png')
plt.show();


meteobydate[['TN', 'TX', 'ETP', 'RR', 'GLOT']].corr(method = "pearson")
from scipy.stats import pearsonr
#dropna nécessaire
pearsonr(meteobydate.TN, meteobydate.TX) # corrélation
pearsonr(meteobydate.TN.tolist(), meteobydate.ETP.tolist()) # corrélation
pearsonr(meteobydate.TN, meteobydate.RR) # corrélation

plt.figure(figsize = (10, 10))
sns.heatmap(data =  meteobydate[['TN', 'TX', 'ETP', 'RR', 'GLOT']].corr(), annot = True, cmap = "coolwarm")
plt.savefig('./reports/figures/heatmap_parametres_' + periode + '.png')
plt.show();