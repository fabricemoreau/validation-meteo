# %%
'''
Script pour extraire un sous-ensemble du jeu de données, qui est trop gros pour être chargé intégralement
'''
# %% [markdown]
#  # Chargement des paquets
#  DuckDB permet de requêter le CSV sans consommer trop de mémoire

# %%
import pandas as pd
import duckdb
from pathlib import Path

# %%
dataPath = Path("..") / "data" / "raw" 
# %% [markdown]
# On retire les stations météo qui ne sont pas dans le jeu de données
# %%
liste_stationspresentes = duckdb.sql("SELECT DISTINCT codearvalis FROM '" + str(dataPath) + "/donneesmeteo_2010-2024_completes.csv'").to_df().codearvalis.astype('int').tolist()
print("nombre de stations présentes", len(liste_stationspresentes))
stations_presentes = pd.read_csv(dataPath / "stationsmeteo.csv", sep=';')
stations_presentes = stations_presentes[stations_presentes.Station.isin(liste_stationspresentes)]
stations_presentes.to_csv(str(dataPath / 'stationsmeteo_2010-2024.csv'), sep=';')

# %% [markdown]
# On prend un échantillon d'1/5 des stations 
# %%
stations_echantillon360 = stations_presentes.sample(363,  random_state = 42)
stations_echantillon360.to_csv(str(dataPath / 'stationsmeteo_363.csv'), sep=';')

# %% [markdown]
# On enregistre les données pour cet échantillon
# %%
query = """
    SELECT * 
    FROM '""" + str(dataPath) + """/donneesmeteo_2010-2024_completes.csv'
    WHERE libellecourt IN ['TN', 'TX', 'TM', 'RR', 'GLOT', 'ETP'] 
          AND codearvalis IN ("""
query += ','.join(map(str, stations_echantillon360.Station.unique().tolist())) + ")"
donnees_363 = duckdb.sql(query)
donnees_363.to_csv(str(dataPath / 'donneesmeteo_2010-2024_363stations.csv'), sep=';', timestamp_format='%Y-%m-%d')

# %% [markdown]
# sélection par groupe d'années: 3 ans
# %%
duree = 3
duckdb.read_csv(dataPath / "donneesmeteo_2010-2024_completes.csv", sep=';')
for annee_deb in range(2010, 2025, duree):
    query = "SELECT * FROM '" + str(dataPath) + "/donneesmeteo_2010-2024_completes.csv' "
    query += "WHERE EXTRACT('YEAR' FROM datemesure) IN ("
    query += ','.join(map(str, list(range(annee_deb, annee_deb + duree, 1))))
    query +=""") 
    AND libellecourt IN ['TN', 'TX', 'TM', 'RR', 'GLOT', 'ETP']
"""
    df = duckdb.query(query)
    df.to_csv(dataPath / 'donneesmeteo_' + str(annee_deb) + '-' + str(annee_deb + duree - 1) + '_completes.csv', sep=';', timestamp_format='%Y-%m-%d')
    liste_stationspresentes = df.to_df().codearvalis.unique().astype('int').tolist()
    
    stations_presentes = pd.read_csv(dataPath / "stationsmeteo.csv", sep=';')
    stations_presentes = stations_presentes[stations_presentes.Station.isin(liste_stationspresentes)]
    stations_presentes.to_csv(str(dataPath / 'stationsmeteo_') + str(annee_deb) + '-' + str(annee_deb + duree - 1) + '.csv', sep=';')
