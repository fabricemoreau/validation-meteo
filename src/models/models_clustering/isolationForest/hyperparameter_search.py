"""
Script de recherche des hyperparamètres de isolationForest
"""
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    make_scorer,    
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
import numpy as np
import plotly.express as px

parameters = ['ETP', 'GLOT', 'TN', 'TX']
SEUIL_ANOMALIE = 0.1
df = pd.read_csv(f"data/processed/meteo_pivot_cleaned_2010-2024_{SEUIL_ANOMALIE}.csv", parse_dates=["datemesure"], sep = ';')
#random_state = 42
#rng = np.random.RandomState(42)

#features = [f"{param}_origine", "day_sin", "day_cos", "year", "Lambert93x", "Lambert93y", "Altitude"]
features = [f"{param_i}_origine" for param_i in parameters]
features = ["day_sin", "day_cos", "Lambert93x", "Lambert93y", "Altitude"] + features

y_train = np.where(df["anomaly"] > 0, -1, 1)
# Get anomalies proportion for contamination
anomalies_proportion = df["anomaly"].mean()
print("anomaly proportion in train:", anomalies_proportion)

# Normalize data for better model performance
scaler = StandardScaler()
train_scaled = scaler.fit_transform(df[features])

# Train Isolation Forest model
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=df["anomaly"].mean(),
    #random_state=random_state,
    n_jobs=-1,
)

contaminations = df["anomaly"].mean() + df["anomaly"].std() * range(4) / 5

del df
log_file = 'data/testModelResults/isolation_forest_total_log.txt'
### gridsearch
param_model = {
    'n_estimators' : [100, 200, 300],
    'contamination' :contaminations,
    'max_samples' :  [0.2, 0.05, 'auto'],
    'max_features' : [8, 6, 4, 1],
}

# a tester ensuite
contaminations = df["anomaly"].mean() + df["anomaly"].std() * range(6) / 2
param_model = {
    'n_estimators' : [25, 50, 75, 100, 200],
    'contamination' : contaminations,
    'max_samples' :  [0.5, 0.2, 0.05],
    'max_features' : [8, 6, 4, 1],
}

param_grid = ParameterGrid(param_model)
accuracy = np.empty(len(param_grid))
precision = np.empty(len(param_grid))
recall = np.empty(len(param_grid))
f1 = np.empty(len(param_grid))


best_recall = 0
y_pred_best = 0
for i in range(0, len(param_grid)):
    print(i, "/", len(param_grid) - 1, ' : ', param_grid[i])
    model = IsolationForest(**(param_grid[i]), n_jobs=-1) # pas de random_state
    model.fit(train_scaled)
    y_pred = model.predict(train_scaled)  # algorithme de clustering
    accuracy[i] = accuracy_score(y_train, y_pred)
    recall[i] = recall_score(y_train, y_pred, pos_label = -1)
    precision[i] = precision_score(y_train, y_pred, pos_label = -1)
    f1[i] = f1_score(y_train, y_pred, pos_label = -1)
    print(classification_report(y_train, y_pred))
    print(
        pd.crosstab(
            y_train, y_pred, rownames=["Classe réelle"], colnames=["Classe prédite"]
        )
    )
    print("recall score ", recall[i])
    print("f1 score ", f1[i])
    print("accuracy score ", accuracy[i])
    if log_file is not None:
        crosstab = pd.crosstab(
                    y_train,
                    y_pred,
                    rownames=["Classe réelle"],
                    colnames=["Classe prédite"],
                )
        f = open(log_file, "a")
        f.write(str(param_grid[i]) + ';' + 
                str(recall[i]) + ';' +
                str(accuracy[i]) + ';' + 
                str(crosstab.iloc[0,0]) + ';' + 
                str(crosstab.iloc[1,1]) + ';' + 
                str(crosstab.iloc[1,0]) + ';' + 
                str(crosstab.iloc[0,1]) + "\n"
                )
        f.close()
        """
        f = open(log_file, "a")
        f.writelines
        f.write("=======================\n")
        f.write("hyperparamètres = " + str(param_grid[i]) + "\n")
        f.write(
            str(
                pd.crosstab(
                    y_train,
                    y_pred,
                    rownames=["Classe réelle"],
                    colnames=["Classe prédite"],
                )
            )
            + "\n"
        )
        f.write("recall score " + str(recall[i]))
        f.write("f1 score " + str(f1[i]))
        f.write("accuracy score " + str(accuracy[i]))
        f.write(str(classification_report(y_train, y_pred)) + "\n")
        f.write("roc = " + str(roc_auc_score(y_train, y_pred)) + "\n")
        f.write("mcc = " + str(matthews_corrcoef(y_train, y_pred)) + "\n")
        f.close()
        """
    if recall[i] > best_recall:
        y_pred_best = y_pred

recap = pd.DataFrame(
        {"param": param_grid, "accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}
    )
print(recap)
recap.to_csv(str(log_file).replace('.txt', '.csv'), sep = ';')

# graphique
fig = px.scatter(recap, x="recall", y="accuracy", hover_data ="param")
fig.show()
fig.savefig(str(log_file).replace('.txt', '.png'))

"""
####idée: graphique des forêts
fn=data.feature_names
cn=data.target_names
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')



# This may not the best way to view each estimator as it is small
fn=data.feature_names
cn=data.target_names
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')
"""