from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .plotting import plot_clusters_3d, plot_elbow_and_silhouette


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
    # Normalize Lambert93x, Lambert93y, and Altitude before clustering
    scaler = StandardScaler()
    stations_df_pos_normalized = scaler.fit_transform(
        stations_df[["Lambert93x", "Lambert93y", "Altitude"]]
    )

    # Cluster stations using Lambert93x, Lambert93y, and Altitude
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
            columns=["Station", "Lambert93x", "Lambert93y", "Altitude", "cluster"],
            errors="ignore",
        )
        .merge(
            stations_df[["Station", "Lambert93x", "Lambert93y", "Altitude", "cluster"]],
            left_on="codearvalis",
            right_on="Station",
            how="left",
        )
        .drop(columns=["Station"])
    )
    return df

def preprocessing(
    df: pd.DataFrame,
    stationsmeteopath: PosixPath,
    parameters: list,
    joinspatial: bool,
    random_state: int = None,
):
    """
    Main function to call for preprocessing
    Parameters:
    df : DataFrame with the meteorological data
    stationsmeteopath:
    parameters : list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX')
    joinsspatial: Join also the spatial information
    random_state:
    """
    if joinspatial:
        # Check if df has been already preprocessed with spatial data
        if not all(
            spatialcolumn in df.columns
            for spatialcolumn in ["Lambert93x", "Lambert93y", "Altitude", "cluster"]
        ):
            df = join_spatial_info(df, stationsmeteopath, random_state=random_state)
    return df
