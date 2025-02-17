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
meteodf = meteodf[meteodf.valeurorigine != -999]

# transformation des dates
meteodf['datemesure'] = pd.to_datetime(meteodf.datemesure)

# Il ne faut retenir qu'une valeur par station jour et paramètre. On retient en priorité idmodeobtention le plus petit
meteodf.fillna({'idmodeobtention': 4}, inplace = True) # on prend les na en dernier recours
meteodf = meteodf.sort_values(by = ['datemesure', 'codearvalis', 'libellecourt', 'idmodeobtention'])
meteodf.drop_duplicates(subset =['datemesure', 'codearvalis', 'libellecourt'], keep = 'first', inplace = True)

#idstatut et idmodeobtention
meteodf.reset_index(drop = True, inplace = True)
meteodf =meteodf.drop(columns = ['idstatut', 'idmodeobtention'])

# on pivote pour avoir une ligne par date
meteobydate = meteodf.pivot(index = ['codearvalis', 'datemesure'], columns = 'libellecourt', values = ['valeur', 'valeurorigine'])
meteobydate.reset_index(inplace = True)
meteobydate.to_csv('./data/processed/meteo_2010-2012.csv', sep=';')

##################################### étude paramètres
meteobydate[meteobydate.valeur.TN.isna()]

meteodf[(meteodf.codearvalis == 145) & (meteodf.libellecourt == 'TN') & (meteodf.datemesure == '2010-05-07')]