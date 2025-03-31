import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model 
import visualkeras

DONNEES = "data/processed/meteo_pivot_cleaned_2010-2024_0.1.csv" 
STATIONS = "data/processed/stationsmeteo_processed_2010-2024.csv"
PARAMETRES = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

# logos
logomf = plt.imread('src/streamlit/assets/LOGO_MF.png')
logoarvalis = plt.imread('src/streamlit/assets/logo-arvalis.png')

# chargement mesures
df = pd.read_csv(DONNEES, sep = ';', parse_dates = ['datemesure'])
df.loc[df.anomaly > 0, 'anomaly'] = 1

stations_timetable = df[['datemesure', 'codearvalis', 'TN']]
stations_timetable = stations_timetable.drop_duplicates()
stations_timetable = stations_timetable.pivot(index = 'codearvalis', columns = 'datemesure', values = 'TN')
# on remplace par valeur binaire 0 ou 1
stations_timetable = stations_timetable.notnull().astype(int)
stations_timetable = stations_timetable.fillna(0)

# séries temporelles des stations
fig = plt.figure()
plt.matshow(stations_timetable, cmap = 'gray_r', aspect = 2)
plt.xticks([i for i in range(0, stations_timetable.shape[1]+1, 366)], [stations_timetable.columns[i].year for i in range(0, stations_timetable.shape[1]+1, 366)])
plt.yticks([i for i in range(0, stations_timetable.shape[0]+1, 100)], [stations_timetable.index[i] for i in range(0, stations_timetable.shape[0]+1, 100)])
plt.xlabel("Temps (années)")
plt.ylabel("Numéro de station météo")
plt.savefig('./reports/figures/series_stations_meteo.png')

# description des paramètres
for param in PARAMETRES:
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    ax.boxplot(df[param])
    ax.set_title(f"Variation de {param}")
    ax = plt.subplot(1, 2, 2)
    ax.hist(df[param], bins = 50)
    ax.set_title(f"Distribution de {param}")
    fig.subplots_adjust(bottom=0.15) 
    
    #ax = fig.add_subplot()
    addLogo = OffsetImage(logomf, zoom=0.05)
    addLogo.set_offset((100,15)) 
    ax.add_artist(addLogo)
    #ax = fig.add_subplot()
    addLogo = OffsetImage(logoarvalis, zoom=0.3)
    addLogo.set_offset((200,15)) 
    ax.add_artist(addLogo)
    plt.savefig('./reports/figures/' + param + '.png')
    
# anomalies 
for param in PARAMETRES:
    df_valides = df[df[f"{param}_anomaly"] == 0]
    df_anomalies = df[df[f"{param}_anomaly"] == 1]
    fig = plt.figure(figsize=(20,10))
    plt.scatter(df_valides.datemesure, df_valides[param], label = f"{param} valides", c = "blue")
    #plt.scatter(df_anomalies.datemesure, df_anomalies[f"{param}"], label = f"{param} corrections", c="green")
    plt.scatter(df_anomalies.datemesure, df_anomalies[f"{param}_origine"], label = f"{param} anomalies", c="red")
    plt.legend()
    plt.savefig('./reports/figures/' + param + '_time.png')

# corrections 
for param in PARAMETRES:
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    df_corrections = df[df[f"{param}_anomaly"] == 1]
    ax.boxplot(df_corrections[f"{param}_difference"])
    ax.set_title(f"Corrections de {param}")
    ax = plt.subplot(1, 2, 2)
    ax.hist(df_corrections[f"{param}_difference"], bins = 50)
    ax.set_title(f"Distribution des corrections de {param}")
    fig.subplots_adjust(bottom=0.15) 
    plt.savefig('./reports/figures/' + param + '_corrections.png')

# days_sin_cos
jours = df.loc[(df.codearvalis == 9151) & df.datemesure.dt.year.isin([2014, 2015]), ['datemesure', 'day_sin', 'day_cos']]
jours['numero_jour'] = jours.datemesure.dt.day_of_year
jours.to_csv('reports/streamlit-data/jours.csv', index = False, sep = ';')

# génération tableaux données
brut = pd.read_csv('data/raw/donneesmeteo_2010-2024_363stations.csv', sep = ';', parse_dates = ['datemesure'])
brut.sort_values(['datemesure', 'codearvalis']).head(50).to_csv('reports/streamlit-data/df_brut.csv', index = False, sep =';')

sample = pd.read_csv('data/processed/meteo_pivot_2010-2012.csv', sep = ';', parse_dates = ['datemesure'])
sample.sort_values(['datemesure', 'codearvalis']).head(10).to_csv('reports/streamlit-data/df_sample.csv', index = False, sep =';')

# chargement liste stations
stations = pd.read_csv(STATIONS, sep = ';')
dates_presence =  df.groupby('codearvalis').datemesure.agg(['min', 'max'])
stations = stations.merge(dates_presence, left_on = 'Station', right_index = True)
stations.to_csv('reports/streamlit-data/stations.csv', index = False, sep = ';')

# schema autoencodeur
model = load_model('models/autoencodeur.keras')
visualkeras.layered_view(model, 
                         to_file='reports/figures/autoencodeur_schema.png',
                         legend = True,
                         draw_volume = False,
                         scale_xy= 2)

"""
from imblearn.under_sampling import RandomUnderSampler

# Définir les caractéristiques (X) et la cible (y)
X = df.drop(columns=['anomaly'])  # Toutes les colonnes sauf "anomaly"
y = df['anomaly']  # Colonne cible

# Initialiser le sous-échantillonneur
rus = RandomUnderSampler()

# Appliquer le sous-échantillonnage
X_resampled, y_resampled = rus.fit_resample(X, y)

# Reconstituer un DataFrame équilibré
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Afficher la répartition des classes après sous-échantillonnage
print(df_resampled['anomaly'].value_counts())
df_resampled.to_csv('reports/streamlit-data/df.csv', index = False, sep = ';')
"""