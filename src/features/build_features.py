'''
Transformations de données communes à tous les usages
'''
import numpy as np
import pandas as pd

periode = '2010-2024'
fichier = './data/processed/meteo_pivot_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
stationstotales = pd.read_csv('./data/raw/stationsmeteo_' + periode + '.csv', sep = ';')

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']


# gestion des données manquantes
print(meteobydate.isna().sum())
# quelques données manquantes. On liste les stations concernées
station_donnees_manquantes = meteobydate[meteobydate[parametres].isna().sum(axis = 1) > 0].codearvalis.unique().tolist()
# on supprime ces stations: on a beaucoup d'autres données
meteobydate = meteobydate[~meteobydate.codearvalis.isin(station_donnees_manquantes)]
stationstotales = stationstotales[~stationstotales.Station.isin(station_donnees_manquantes)]

# Enrichissement des dates
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure)
meteobydate['jourjulien'] = meteobydate.datemesure.dt.dayofyear
meteobydate['mois'] = meteobydate.datemesure.dt.month
# saison simplifiée: hiver = janvier, février, mars...
meteobydate['saison'] = (meteobydate['mois'] - 1) // 3

## ajout de la colonne anomalie, on complète les données d'origine identiques aux données corrigées
for param in parametres:
    meteobydate[param + '_difference'] = (meteobydate[param + '_origine'] - meteobydate[param])
    meteobydate[param + '_anomalie'] = (np.abs(meteobydate[param + '_difference']) > 0).astype(int)
    # on complète les na valeur d'origine par valeur validée seulement en cas d'anomalie
    meteobydate.loc[meteobydate[param + '_anomalie'] == 1, param + '_origine'] = meteobydate.loc[meteobydate[param + '_anomalie'] == 1, param]
    # si "_origine" est à na, il y a une anomalie même si on ne connait pas la valeur d'origine, mais on laisse à na la valeur d'origine
    meteobydate.loc[meteobydate[param + '_origine'].isna(), param + '_anomalie'] = 1
    
# au moins une anomalie sur la ligne
meteobydate['anomalie'] = meteobydate[[s + '_anomalie' for s in parametres]].sum(axis = 1)

# classification des pluies
#meteobydate['pluiepresente'] = (meteobydate.RR > 0)
#meteobydate['pluiepresente_origine'] = (meteobydate.RR_origine > 0)
# on pourrait classifier par quantiles, mais on construit des classes par rapport à l'usage de la donnée et 
pluie_intervalles = pd.IntervalIndex.from_tuples([(-1, 0), (0, 0.5), (0.5, 5), (5, 15), (15, 50), (50, 500)], 
                                                 name = ['paspluie', 'tresfaible', 'faible', 'moyen', 'fort', 'extreme'])
meteobydate['pluieclassif'] = pd.cut(meteobydate.RR, pluie_intervalles)
meteobydate['pluieclassif'] = meteobydate.pluieclassif.cat.rename_categories(pluie_intervalles.name)
meteobydate['pluieclassif_origine'] = pd.cut(meteobydate.RR_origine, pluie_intervalles)
meteobydate['pluieclassif_origine'] = meteobydate.pluieclassif_origine.cat.rename_categories(pluie_intervalles.name)

meteobydate = meteobydate.merge(stationstotales[['Station', 'Altitude', 'Lambert93x', 'Lambert93y']], left_on = 'codearvalis', right_on = "Station").drop(columns = 'Station')

meteobydate.to_csv('./data/processed/meteo_pivot_cleaned_' + periode + '.csv', sep = ';', index = False)
