from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def isolationForest(df: pd.DataFrame, parameters:list):
    """
    Apply isolation forest algorithm on each parameter separately.
    Parameters:
    df: dataframe to apply the algorithm
    parameters: list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX') 
    """
    for param in tqdm(parameters, desc="Training Isolation Forests"):
        print("training", param)
        features = [f"{param}_origine", "month_sin", "month_cos", "Latitude", "Longitude", "Altitude"]
        # Split data into train and test sets
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        df.loc[test_data.index, "is_test"] = 1  # Mark test rows
        # Get anomalies proportion for contamination
        anomalies_proportion = df[f"{param}_anomaly"].sum() / df.shape[0]

        # Normalize data for better model performance
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[features])
        test_scaled = scaler.transform(test_data[features])
        # Train Isolation Forest model
        iso_forest = IsolationForest(
            n_estimators=100, contamination=anomalies_proportion, random_state=42
        )
        iso_forest.fit(train_scaled)
        # Predict on test set
        df.loc[test_data.index, f"{param}_anomaly_score"] = iso_forest.predict(
            test_scaled
        )

        # Identify anomalies for the specific parameter
        df.loc[test_data.index, f"{param}_is_anomaly"] = df.loc[
            test_data.index, f"{param}_anomaly_score"
        ].apply(lambda x: 1 if x == -1 else 0)

        print(
            pd.crosstab(
                df[f"{param}_anomaly"],
                df[f"{param}_is_anomaly"],
                rownames=["real"],
                colnames=["predicted"],
            )
        )
        accuracy = accuracy_score(
            df.loc[test_data.index, f"{param}_anomaly"],
            df.loc[test_data.index, f"{param}_is_anomaly"],
        )
        print(f"Accuracy for {param}: {accuracy:.4f}")
    return df
