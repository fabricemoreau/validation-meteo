'''
Script pour extraire un sous-ensemble du jeu de données de 5Go
'''
import pandas as pd
import duckdb

# sélection d'un échantillon de stations (1/5 du jeu de données)
liste_stationspresentes = duckdb.sql("SELECT DISTINCT codearvalis FROM './data/raw/donneesmeteo_2010-2024_completes.csv'").to_df().codearvalis.astype('int').tolist()
print("nombre de stations présentes", len(liste_stationspresentes))
stations_presentes = pd.read_csv("./data/raw/stationsmeteo.csv", sep=';')
stations_presentes = stations_presentes[stations_presentes.Station.isin(liste_stationspresentes)]
stations_presentes.to_csv('./data/raw/stationsmeteo_2010-2024.csv', sep=';')
# on prend 1/5 des stations =363 pour avoir un jeu de données d'environ 600Mo
stations_echantillon360 = stations_presentes.sample(363,  random_state = 42)
stations_echantillon360.to_csv('./data/raw/stationsmeteo_363.csv', sep=';')

query = """
    SELECT * 
    FROM './data/raw/donneesmeteo_2010-2024_completes.csv'
    WHERE libellecourt IN ['TN', 'TX', 'TM', 'RR', 'GLOT', 'ETP'] 
          AND codearvalis IN ("""
query += ','.join(map(str, stations_echantillon360.Station.unique().tolist())) + ")"
donnees_363 = duckdb.sql(query)
donnees_363.to_csv('./data/raw/donneesmeteo_2010-2024_363stations.csv', sep=';', timestamp_format='%Y-%m-%d')

# sélection par groupe d'années: 3 ans
duree = 3
duckdb.read_csv("./data/raw/donneesmeteo_2010-2024_completes.csv", sep=';')
for annee_deb in range(2010, 2025, duree):
    query = """
    SELECT * 
FROM './data/raw/donneesmeteo_2010-2024_completes.csv'
WHERE EXTRACT('YEAR' FROM datemesure) IN ("""
    query += ','.join(map(str, list(range(annee_deb, annee_deb + duree, 1))))
    query +=""") 
    AND libellecourt IN ['TN', 'TX', 'TM', 'RR', 'GLOT', 'ETP']
"""
    df = duckdb.query(query)
    df.to_csv('./data/raw/donneesmeteo_' + str(annee_deb) + '-' + str(annee_deb + duree - 1) + '_completes.csv', sep=';', timestamp_format='%Y-%m-%d')
    liste_stationspresentes = df.to_df().codearvalis.unique().astype('int').tolist()
    
    stations_presentes = pd.read_csv("./data/raw/stationsmeteo.csv", sep=';')
    stations_presentes = stations_presentes[stations_presentes.Station.isin(liste_stationspresentes)]
    stations_presentes.to_csv('./data/raw/stationsmeteo_' + str(annee_deb) + '-' + str(annee_deb + duree - 1) + '.csv', sep=';')
