'''
Script pour refactoriser données depuis extract_subdata
'''
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin  # Importation du mixin TransformerMixi
from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(display="diagram")
#set_config(transform_output="pandas")

class DropDuplicated(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Il ne faut retenir qu'une valeur par station jour et paramètre. On retient en priorité idmodeobtention le plus petit
        X = X.fillna({'idmodeobtention': 4}) # on prend les na en dernier recours
        X = X.sort_values(by = ['datemesure', 'codearvalis', 'libellecourt', 'idmodeobtention'])
        X = X.drop_duplicates(subset =['datemesure', 'codearvalis', 'libellecourt'], keep = 'first')
        return X
        
class ToDate(TransformerMixin):
    # pourra être modifié pour ajouter un seuil de correction
    def __init__(self):
        return None
    
    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X['datemesure'] = pd.to_datetime(X.datemesure)
        return X

class Cleanup(TransformerMixin):
    # pourra être modifié pour ajouter un seuil de correction
    '''parameters_to_keep: array of meteo parameter to keep in the dataframe'''
    def __init__(self, parameters_to_keep):
        self.parameters_to_keep = parameters_to_keep
        return None
    
    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X = X.loc[X.libellecourt.isin(self.parameters_to_keep)]
        X.loc[X.valeurorigine.isna(), 'valeurorigine'] = X.loc[X.valeurorigine.isna(), 'valeur'] 
        X = X.replace(to_replace= {"valeurorigine" : -999}, value=np.nan)
        X = X.drop(columns = ['idstatut', 'idmodeobtention'])
        return X

class PivotTable(TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters
        return None
    
    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X = X.pivot(index = ['codearvalis', 'datemesure'], columns = 'libellecourt', values = self.parameters)
        return X

class ArrangePivot(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X : pd.DataFrame, y = None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        valeurorigine_df = X.valeurorigine
        valeurorigine_df = valeurorigine_df.rename(columns = {'ETP': 'ETP_origine', 'GLOT': 'GLOT_origine', 'RR': 'RR_origine', 'TN': 'TN_origine', 'TX': 'TX_origine'})
        fusion = pd.concat([X.valeur, valeurorigine_df], axis = 1)
        fusion = fusion.reset_index()
        return fusion

pivot_pipeline = Pipeline(steps = [('Suppression des doublons', DropDuplicated()),
                                   ('Conversion des dates', ToDate()),
                                   ('Nettoyage des données', Cleanup(['ETP', 'GLOT', 'RR', 'TN', 'TX'])),
                                   ('Pivot', PivotTable(['valeur', 'valeurorigine'])),
                                   ('Arrange Pivot', ArrangePivot())
                                   ],
                  verbose = True)


#periode = '2010-2012'
#fichier = './data/raw/donneesmeteo_' + periode + '_completes.csv'
#meteodf = pd.read_csv(fichier, sep = ';', index_col = False)
#pivot_meteodf = pivot_pipeline.fit_transform(meteodf)

duree = 3
df_total = pd.DataFrame()
for annee_deb in range(2010, 2025, duree):
    periode = str(annee_deb) + '-' + str(annee_deb + duree - 1)
    print(periode)
    fichier = './data/raw/donneesmeteo_' + periode + '_completes.csv'
    df = pd.read_csv(fichier, sep = ';')
    pivot_meteodf = pivot_pipeline.fit_transform(df)
    pivot_meteodf.to_csv('./data/processed/meteo_pivot_' + periode + '.csv', sep=';', index = False)
    df_total = pd.concat([df_total, pivot_meteodf], axis = 0)

df_total.to_csv('./data/processed/meteo_pivot_2010-2024.csv', sep=';', index = False)
