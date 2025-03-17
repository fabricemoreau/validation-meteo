from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from plotting import plot_clusters_3d, plot_elbow_and_silhouette


def join_spatial_info(
    df: pd.DataFrame, spatial_data_path: Path, plots: bool = True, random_state=None
):
    """
    Merges the spatial information with the meteorological one
    Parameters:
    df: meteo dataset.
    spatial_data_path: path to the dataset with the information per station
    random_state : set random_state of sklearn function (for reproducibility)
    """
    print(f"reading spatial data from {spatial_data_path}")
    stations_df = pd.read_csv(spatial_data_path, sep=";")
    print(stations_df.head())
    # Normalize Latitude, Longitude, and Altitude before clustering
    scaler = StandardScaler()
    stations_df_pos_normalized = scaler.fit_transform(
        stations_df[["Latitude", "Longitude", "Altitude"]]
    )

    # Cluster stations using Latitude, Longitude, and Altitude
    if plots:
        plot_elbow_and_silhouette(stations_df_pos_normalized, 20)

    num_clusters = 4  # You can adjust this based on the elbow method or silhouette
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    stations_df["cluster"] = kmeans.fit_predict(stations_df_pos_normalized)
    plot_clusters_3d(stations_df)
    # Merge station data into the meteorological dataset
    # drop columns before merge to avoid duplicated
    df = (
        df.drop(
            columns=["Station", "Latitude", "Longitude", "Altitude", "cluster"],
            errors="ignore",
        )
        .merge(
            stations_df[["Station", "Latitude", "Longitude", "Altitude", "cluster"]],
            left_on="codearvalis",
            right_on="Station",
            how="left",
        )
        .drop(columns=["Station"])
    )
    return df


def preprocessing(df: pd.DataFrame, parameters: list):
    """
    Pre-process the dataset before training.

    Parameters:
    df : DataFrame with the meteorological data
    parameters : list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX')
    """

    # Extract date-based features
    df["month"] = df["datemesure"].dt.month
    # Encode cyclic nature of months
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Initialize test set marker
    df["is_test"] = 0

    # Create separate anomaly columns based on human corrections, using a dynamic threshold
    # TODO: get out this function
    def detect_real_anomaly(row, param, std_threshold):
        threshold = std_threshold[param]  # Use parameter-specific threshold
        return 1 if abs(row[param] - row[f"{param}_origine"]) > threshold else 0

    # Compute standard deviation-based thresholds for each parameter
    std_threshold = {
        param: df[f"{param}_origine"].std() * 0.1 for param in parameters
    }  # 10% of std deviation

    for param in tqdm(parameters, desc="Preprocessing, finding real anomalies"):
        tqdm.pandas(desc=f"{param} anomalies")
        df[f"{param}_anomaly"] = df.progress_apply(
            lambda row: detect_real_anomaly(row, param, std_threshold), axis=1
        )
        anomaliesProportion = df[f"{param}_anomaly"].sum() / df.shape[0]
        print(f"{param} anomalies percentage: {anomaliesProportion * 100:.3}")

    return df
