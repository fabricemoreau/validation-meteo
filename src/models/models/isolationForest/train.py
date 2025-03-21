from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
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

    # Split data into train and test sets
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=random_state
    )
    df.loc[test_data.index, "is_test"] = 1  # Mark test rows

    for param in tqdm(parameters, desc="Training Isolation Forests"):
        print("training", param)
        features = [f"{param}_origine", "month_sin", "month_cos"]
        if spatial_info:
            features.append("cluster")

        # Get anomalies proportion for contamination
        anomalies_proportion = train_data[f"{param}_anomaly"].sum() / train_data.shape[0]
        print("anomaly proportion in train:", anomalies_proportion)
        print("anomaly proportion in test:", test_data[f"{param}_anomaly"].sum() / test_data.shape[0])

        # Normalize data for better model performance
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[features])
        test_scaled = scaler.transform(test_data[features])
        # Train Isolation Forest model
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=anomalies_proportion,
            random_state=random_state,
            n_jobs=-1,
        )
        iso_forest.fit(train_scaled)
        # Predict on test set
        df.loc[test_data.index, f"{param}_anomaly_score"] = iso_forest.predict(
            test_scaled
        )

        # Identify anomalies for the specific parameter
        df.loc[test_data.index, f"{param}_anomaly_pred"] = df.loc[
            test_data.index, f"{param}_anomaly_score"
        ].apply(lambda x: 1 if x == -1 else 0)

        print(
            pd.crosstab(
                df[f"{param}_anomaly"],
                df[f"{param}_anomaly_pred"],
                rownames=["real"],
                colnames=["predicted"],
            )
        )
        accuracy = accuracy_score(
            df.loc[test_data.index, f"{param}_anomaly"],
            df.loc[test_data.index, f"{param}_anomaly_pred"],
        )
        precision = precision_score(
            df.loc[test_data.index, f"{param}_anomaly"],
            df.loc[test_data.index, f"{param}_anomaly_pred"],
        )
        recall = recall_score(
            df.loc[test_data.index, f"{param}_anomaly"],
            df.loc[test_data.index, f"{param}_anomaly_pred"],
        )
        print(
            f"{param} -> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
        )
        print(
            classification_report(
                df.loc[test_data.index, f"{param}_anomaly"],
                df.loc[test_data.index, f"{param}_anomaly_pred"],
            )
        )

    return df
