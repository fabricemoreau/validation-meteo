from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm


def train(
    df: pd.DataFrame,
    parameters: list,
    spatial_info: bool = False,
    log_file: str = None,
    random_state: int =None,
):
    """
    Apply isolation forest algorithm on each parameter separately.

    Parameters:
    df: dataframe to apply the algorithm
    parameters: list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX')
    random_state: to set random_state of sklearn functions (for reproducibility)
    """
    # Check if the df has been already preprocessed
    if not all(f"{param}_anomaly" in df.columns for param in parameters):
        raise Exception(
            "Columns  '<param>_anomaly' missing in the database, please input a file built with command 'prepare'"
        )
    if spatial_info:
        # Check if df has been already preprocessed with spatial data
        if not all(
            spatialcolumn in df.columns
            for spatialcolumn in ["Lambert93x", "Lambert93y", "Altitude", "cluster"]
        ):
            raise Exception(
                "Spatial Columns missing in the database, please input a file built with command 'prepare'"
            )

    df["is_test"] = 1  # Mark test rows
    
    print("global isolation forest")
    # global
    features = [f"{param_i}_origine" for param_i in parameters]
    features = ["day_sin", "day_cos", "Lambert93x", "Lambert93y", "Altitude"] + features
    df["anomaly"] = np.where(df["anomaly"] > 0, 1, 0)
    # Normalize data for better model performance
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df[features])

    # Train Isolation Forest model: hyperparameters set from search
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.14,
        max_features = 8,
        max_samples = 0.2,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_forest.fit(train_scaled)
    # Predict on test set
    df["anomaly_score"] = iso_forest.predict(
        train_scaled
    )

    # Identify anomalies for the specific parameter
    df["anomaly_pred"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)
    
    print(
        pd.crosstab(
            df["anomaly"],
            df["anomaly_pred"],
            rownames=["real"],
            colnames=["predicted"],
        )
    )
    accuracy = accuracy_score(
        df["anomaly"],
        df["anomaly_pred"],
    )
    precision = precision_score(
        df["anomaly"],
        df["anomaly_pred"],
    )
    recall = recall_score(
        df["anomaly"],
        df["anomaly_pred"],
    )
    print(
        f"global-> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
    )
    print(
        classification_report(
            df["anomaly"],
            df["anomaly_pred"],
        )
    )


    # par paramètres: pas d'hyperparamètre optimisé
    for param in tqdm(parameters, desc="Training Isolation Forests"):
        print("training", param)
        features = [f"{param}_origine", "day_sin", "day_cos"]
        if spatial_info:
            features.append("cluster")

        # Get anomalies proportion for contamination
        anomalies_proportion = df[f"{param}_anomaly"].sum() / df.shape[0]
        print("anomaly proportion in train:", anomalies_proportion)

        # Normalize data for better model performance
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(df[features])
        # Train Isolation Forest model
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=anomalies_proportion,            
            random_state=random_state,
            n_jobs=-1,
        )
        iso_forest.fit(train_scaled)
        # Predict on test set
        df[f"{param}_anomaly_score"] = iso_forest.predict(
            train_scaled
        )

        # Identify anomalies for the specific parameter
        df[f"{param}_anomaly_pred"] = df[f"{param}_anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

        print(
            pd.crosstab(
                df[f"{param}_anomaly"],
                df[f"{param}_anomaly_pred"],
                rownames=["real"],
                colnames=["predicted"],
            )
        )
        accuracy = accuracy_score(
            df[f"{param}_anomaly"],
            df[f"{param}_anomaly_pred"],
        )
        precision = precision_score(
            df[f"{param}_anomaly"],
            df[f"{param}_anomaly_pred"],
        )
        recall = recall_score(
            df[f"{param}_anomaly"],
            df[f"{param}_anomaly_pred"],
        )
        print(
            f"{param} -> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
        )
        print(
            classification_report(
                df[f"{param}_anomaly"],
                df[f"{param}_anomaly_pred"],
            )
        )

    return df
