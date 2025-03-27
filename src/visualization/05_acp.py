""" ACP sur les données """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

periode = '2010-2024'
SEUIL_ANOMALIE = 0.1
fichier = f"./data/processed/meteo_pivot_cleaned_{periode}_{SEUIL_ANOMALIE}.csv"
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = ['datemesure'])
meteobydate = meteobydate.dropna()

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']


y = np.where(meteobydate.anomaly > 0, 1, 0)
anomalies = meteobydate[[s + '_anomaly' for s in parametres]]

col_to_keep = [s + '_origine' for s in parametres]
col_to_keep.extend(['Altitude', 'Lambert93x', 'Lambert93y', 'day_cos', 'day_sin'])
df = meteobydate[col_to_keep]


sc = StandardScaler()
df_normalise =sc.fit_transform(df)
df_normalise = pd.DataFrame(df_normalise, columns = df.columns)
# df_normalise joue le rôle de X

plt.figure(figsize = (15, 15))
sns.heatmap(df_normalise.corr(), annot=True, cmap='viridis');
plt.savefig('./reports/figures/acp_heatmap.png')
#plt.show();

pca = PCA()
Coord = pca.fit_transform(df_normalise)
print('Part ratio de  variance expliquée :', pca.explained_variance_ratio_)
plt.figure()
plt.xlabel('Nombre de composantes')
plt.ylabel('Part de variance expliquée')
plt.axhline(y = 0.9, color ='r', linestyle = '--')
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title(f"Part de variance expliquée - seuil anomalie {SEUIL_ANOMALIE}")
plt.savefig('./reports/figures/acp_variance_expliquee.png')
#plt.show();
print("La part de variance expliquée est", round(pca.explained_variance_ratio_.sum(),2))

L1 = list(pca.explained_variance_ratio_[0:4])
L1.append(sum(pca.explained_variance_ratio_[4:]))
plt.figure()
plt.pie(L1, labels=['PC1', 'PC2', 'PC3', 'PC4', 'Autres'], 
        autopct='%1.3f%%')
plt.title(f"Part de variance expliquée - seuil anomalie {SEUIL_ANOMALIE}")
plt.savefig('./reports/figures/acp_variance_expliquee_ratio.png')
#plt.show();

pca = PCA(n_components = 2)
data_2D = pca.fit_transform(df_normalise)
# graphe 2D
fig = plt.figure(figsize = (10, 10))
scatter = plt.scatter(data_2D[y == 0, 0], data_2D[y == 0, 1], c='blue', label='correcte')
scatter = plt.scatter(data_2D[y == 1, 0], data_2D[y == 1, 1], c='red', label='anomalie')
# Ajout des labels et du titre
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f"Données projetées sur les 2 axes de l'ACP - seuil anomalie {SEUIL_ANOMALIE}")
plt.legend()
plt.savefig('./reports/figures/acp_graphe2d.png')
#plt.show();


# graphe 2D par paramètre
for param in parametres:
    y = meteobydate[f"{param}_anomaly"]
    fig = plt.figure(figsize = (10, 10))
    scatter = plt.scatter(data_2D[y == 0, 0], data_2D[y == 0, 1], c='blue', label='correcte')
    scatter = plt.scatter(data_2D[y == 1, 0], data_2D[y == 1, 1], c='red', label='anomalie')
    # Ajout des labels et du titre
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f"Données projetées sur les 2 axes de l'ACP pour {param} - seuil anomalie {SEUIL_ANOMALIE}")
    plt.legend()
    plt.savefig(f"./reports/figures/acp_graphe2d_{param}.png")
    #plt.show();

