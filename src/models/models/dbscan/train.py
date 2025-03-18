"""Clustering: DBSCAN"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)
def train(
    df: pd.DataFrame,
    parameters: list,
    spatial_info: bool = False,
    log_file: str = None,
    random_state: int = None,
):
    """
        Apply DBScan algorithm on every parameter

    Parameters:
    df: dataframe to apply the algorithm
    parameters: list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX')
    random_state: to set random_state of sklearn functions (for reproducibility)
    """
    # on travaille sur un sous-ensemble du jeu de données pour plus de rapidité
    df = df.sample(frac = 0.2, random_state = random_state)
    col_to_keep = [s + "_origine" for s in parameters]
    col_to_keep.extend(
        ["Altitude", "Lambert93x", "Lambert93y", "month_sin", "month_cos"]
    )
    y_total = (df.anomalie > 1).astype(int)

    X = df[col_to_keep]
    y = y_total

    # X = X.astype({'datemesure': 'int64'})

    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns=X.columns)

    # recherche sur quelques combinaisons d'hyperparamètres
    param_model = {
        "eps": [0.6, 0.7, 0.8],
        "min_samples": [6, 5, 4],
        "metric": ["euclidean"],
    }
    param_model = {"eps": [0.6], "min_samples": [6]}

    param_grid = ParameterGrid(param_model)
    accuracy = np.empty(len(param_grid))
    roc = np.empty(len(param_grid))
    mcc = np.empty(len(param_grid))

    best_f1_score = 0
    y_pred_best = 0
    for i in range(0, len(param_grid)):
        print(param_grid[i])
        model = DBSCAN(**(param_grid[i]), n_jobs=-1)
        model.fit(X_norm.values)
        y_pred = model.labels_  # algorithme de clustering
        y_pred[y_pred >= 1] = 0
        y_pred[y_pred == -1] = 1
        accuracy[i] = accuracy_score(y, y_pred)
        roc[i] = roc_auc_score(y, y_pred)
        mcc[i] = matthews_corrcoef(y, y_pred)
        print(classification_report(y, y_pred))
        print(
            pd.crosstab(
                y, y_pred, rownames=["Classe réelle"], colnames=["Classe prédite"]
            )
        )
        print("recall class")
        print("MCC Score", matthews_corrcoef(y, y_pred))
        if log_file is not None:
            f = open(log_file, "a")
            f.writelines
            f.write("=======================\n")
            f.write("hyperparamètres = " + str(param_grid[i]) + "\n")
            f.write(
                str(
                    pd.crosstab(
                        y,
                        y_pred,
                        rownames=["Classe réelle"],
                        colnames=["Classe prédite"],
                    )
                )
                + "\n"
            )
            f.write(str(classification_report(y, y_pred)) + "\n")
            f.write("roc = " + str(roc_auc_score(y, y_pred)) + "\n")
            f.write("mcc = " + str(matthews_corrcoef(y, y_pred)) + "\n")
            f.close()
        le_f1_score = f1_score(y, y_pred, average=None)
        if le_f1_score[1] > best_f1_score:
            y_pred_best = y_pred

    print(
        pd.DataFrame(
            {"param": param_grid, "accuracy": accuracy, "roc": roc, "mcc": mcc}
        )
    )
    df["is_test"] = 1
    df["anomaly_pred"] = y_pred_best
    # on reporte les anomalies à tous les paramètres
    for param in parameters:
        df[param + '_anomaly'] = df[param + '_anomalie'] # a améliorer
        df[param + '_anomaly_pred'] = y_pred_best
    return df


# plantage pour eps > 0.6
# moins pire modèle pour
# param_model = { 'eps': [0.2, 0.3, 0.4, 0.5], 'min_samples' : [6, 5, 4] }
# hyperparamètres = {'min_samples': 6, 'eps': 0.3}
# Classe prédite        0        1
# Classe réelle
# 0               2391835  2136342
# 1                 31547    34718
#               precision    recall  f1-score   support

#            0       0.99      0.53      0.69   4528177
#            1       0.02      0.52      0.03     66265

#     accuracy                           0.53   4594442
#    macro avg       0.50      0.53      0.36   4594442
# weighted avg       0.97      0.53      0.68   4594442

# roc = 0.5260690607902315
# mcc = 0.012451232049547315
# moins pire modèle pour
# param_model = { 'eps': [0.2, 0.3, 0.4, 0.5], 'min_samples' : [100, 10, 5] }
# hyperparamètres = {'min_samples': 100, 'eps': 0.5}
# Classe prédite        0        1
# Classe réelle
# 0               2766613  1761564
# 1                 36802    29463
#              precision    recall  f1-score   support
#
#           0       0.99      0.61      0.75   4528177
#           1       0.02      0.44      0.03     66265
#
#     accuracy                           0.61   4594442
#    macro avg       0.50      0.53      0.39   4594442
# weighted avg       0.97      0.61      0.74   4594442
#
# roc = 0.5278005399124528
# mcc = 0.013592278410455095


"""
model = DBSCAN(eps = 0.3, min_samples = 4, n_jobs = -1)
model.fit(X_norm.values)
y_pred = model.labels_ # algorithme de clustering
y_pred[y_pred >= 1] = 0
y_pred[y_pred == -1] = 1
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
print(pd.crosstab(y, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
print('roc', roc_auc_score(y, y_pred))
print('MCC Score', matthews_corrcoef(y, y_pred))
"""

## questions
# y_train?
# git: stocker figures?

## pistes
# définir calcul de distance (poids altitude temps)
# retirer pluie? évaluer par rapport à un seul paramètre météo
# autres hyperparamètres: algorithm,
# segmenter jeux de données par année? avec une fenêtre?
