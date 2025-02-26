'''
ACP sur les données
'''

#%matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

periode = '2010-2024'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
fichier = './data/processed/meteo_pivot_cleaned_time_space_2010-2024.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')

col_to_keep = [s + '_origine' for s in parametres]
for i in range(1, 6):
    col_to_keep.extend([s + '_' + str(i) for s in parametres])
col_to_keep.extend(['Altitude', 'Lambert93x', 'Lambert93y', 'datemesure', 'jourjulien', 'mois', 'saison', 'pluieclassif_origine'])
df = meteobydate[col_to_keep]

pluie_enc = OneHotEncoder(sparse_output = False)
pluies = pd.DataFrame(pluie_enc.fit_transform(df[['pluieclassif_origine']]), columns = pluie_enc.get_feature_names_out())
df = pd.concat([df.drop(columns = 'pluieclassif_origine'), pluies], axis = 1)

del pluies
del meteobydate


sc = StandardScaler()
df_normalise =sc.fit_transform(df.drop(columns = "datemesure"))
df_normalise = pd.DataFrame(df_normalise, columns = df.drop(columns = "datemesure").columns)


plt.figure(figsize = (30, 30))
sns.heatmap(df_normalise.corr(), annot=True, cmap='viridis');
plt.savefig('./reports/figures/acp_heatmap.png')
plt.show();

pca = PCA()
Coord = pca.fit_transform(df_normalise)
print('Part ratio de  variance expliquée :', pca.explained_variance_ratio_)
plt.figure()
plt.xlabel('Nombre de composantes')
plt.ylabel('Part de variance expliquée')
plt.axhline(y = 0.9, color ='r', linestyle = '--')
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.savefig('./reports/figures/acp_variance_expliquee.png')
plt.show();


L1 = list(pca.explained_variance_ratio_[0:10])
L1.append(sum(pca.explained_variance_ratio_[10:]))
plt.figure()
plt.pie(L1, labels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'Autres'], 
        autopct='%1.3f%%')
plt.savefig('./reports/figures/acp_variance_expliquee_ratio.png')
plt.show();

# cercle des corrélations
racine_valeurs_propres = np.sqrt(pca.explained_variance_)
corvar = np.zeros((30, 30))
for k in range(30):
    corvar[:, k] = pca.components_[:, k] * racine_valeurs_propres[k]

# Délimitation de la figure
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

# Affichage des variables
for j in range(30):
    plt.annotate(df_normalise.columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
    plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.6, alpha=0.5, head_width=0.03, color='b')

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Cercle et légendes
cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
axes.add_artist(cercle)
plt.xlabel('AXE 1')
plt.ylabel('AXE 2')
plt.savefig('./reports/figures/acp_cercle.png')
plt.show();
