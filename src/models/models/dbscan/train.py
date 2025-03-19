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
    y_total = (df.anomaly > 1).astype(int)

    X = df[col_to_keep]
    y = y_total

    # X = X.astype({'datemesure': 'int64'})

    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns=X.columns)

    # recherche sur quelques combinaisons d'hyperparamètres
    param_model = {
        "eps": [0.5, 0.6, 0.7],
        "min_samples": [20, 10, 5],
        "metric": ["euclidean", "minkowski"],
    }
    param_model = {"eps": [0.5, 0.45, 0.4], "min_samples": [50, 25, 15], "metric": ["euclidean"]}

    param_grid = ParameterGrid(param_model)
    accuracy = np.empty(len(param_grid))
    precision = np.empty(len(param_grid))
    recall = np.empty(len(param_grid))
    roc = np.empty(len(param_grid))
    mcc = np.empty(len(param_grid))

    best_recall = 0
    y_pred_best = 0
    for i in range(0, len(param_grid)):
        print(param_grid[i])
        model = DBSCAN(**(param_grid[i]), p = 4, n_jobs=-1)
        model.fit(X_norm.values)
        y_pred = model.labels_  # algorithme de clustering
        y_pred[y_pred >= 1] = 0
        y_pred[y_pred == -1] = 1
        accuracy[i] = accuracy_score(y, y_pred)
        recall[i] = recall_score(y, y_pred)
        precision[i] = precision_score(y, y_pred)
        roc[i] = roc_auc_score(y, y_pred)
        mcc[i] = matthews_corrcoef(y, y_pred)
        print(classification_report(y, y_pred))
        print(
            pd.crosstab(
                y, y_pred, rownames=["Classe réelle"], colnames=["Classe prédite"]
            )
        )
        print("recall class", recall_score(y, y_pred))
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
        if recall[i] > best_recall:
            y_pred_best = y_pred

    recap = pd.DataFrame(
            {"param": param_grid, "accuracy": accuracy, "recall": recall, "precision": precision, "roc": roc, "mcc": mcc}
        )
    print(recap)
    recap.to_csv(str(log_file).replace('.txt', '.csv'))
    df["is_test"] = 1
    df["anomaly_pred"] = y_pred_best
    # on reporte les anomalies à tous les paramètres
    for param in parameters:
        df[param + '_anomaly_pred'] = y_pred_best
    return df

