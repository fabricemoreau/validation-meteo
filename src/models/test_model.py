from inspect import getmembers, isfunction
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import train_model
import typer
from typing_extensions import Annotated
from tqdm import tqdm


testModel = typer.Typer()


def preprocessing(df: pd.DataFrame, parameters: list):
    """
    Pre-process the dataset before training.

    Parameters:
    df : DataFrame with the meteorological data
    parameters : list of parameters to consider, (typically 'ETP', 'GLOT', 'RR', 'TN', 'TX')
    """
    # Convert date column to datetime format
    df["datemesure"] = pd.to_datetime(df["datemesure"])

    # Extract date-based features
    df["month"] = df["datemesure"].dt.month
    # Encode cyclic nature of months
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Initialize test set marker
    df["is_test"] = 0

    # Create separate anomaly columns based on human corrections, using a dynamic threshold
    def detect_real_anomaly(row, param, std_threshold):
        threshold = std_threshold[param]  # Use parameter-specific threshold
        return 1 if abs(row[param] - row[f"{param}_origine"]) > threshold else 0

    # Compute standard deviation-based thresholds for each parameter
    std_threshold = {param: df[f"{param}_origine"].std() * 0.1 for param in parameters}  # 10% of std deviation
    
    for param in tqdm(parameters, desc="Preprocessing, finding real anomalies"):
        tqdm.pandas(desc=f"{param} anomalies")
        df[f"{param}_anomaly"] = df.progress_apply(lambda row: detect_real_anomaly(row, param, std_threshold), axis=1)
        anomaliesProportion = df[f"{param}_anomaly"].sum()/df.shape[0]
        print(f"{param} anomalies percentage: {anomaliesProportion * 100:.3}")

    return df


# Function to plot anomalies with Plotly
def plot_anomalies(parameter, df):
    """
    Plots anomaly detection results for a given meteorological parameter using Plotly.

    Parameters:
    parameter (str): The name of the meteorological parameter to visualize (e.g., 'ETP', 'GLOT', 'RR', 'TN', 'TX').
    df (pd.DataFrame): The dataset containing the original values, predicted anomalies, and real anomalies.

    The function creates a scatter plot with:
    - Blue points representing normal values.
    - Green points representing real anomalies (human corrections).
    - Red points representing predicted anomalies (detected by Isolation Forest).
    """
    test_data = df[df["is_test"] == 1]
    test_data["anomaly_type"] = test_data.apply(
        lambda row: "TP"
        if row[f"{parameter}_anomaly"] == 1 and row[f"{parameter}_is_anomaly"] == 1
        else "FN"
        if row[f"{parameter}_anomaly"] == 1
        else "FP"
        if row[f"{parameter}_is_anomaly"] == 1
        else "TN",
        axis=1,
    )
    fig = px.scatter(
        test_data,
        x="datemesure",
        y=f"{parameter}_origine",
        color="anomaly_type",
        color_discrete_map={
            "Normal": "blue",
            "Real Anomaly": "green",
            "Predicted Anomaly": "red",
        },
        title=f"Anomaly Detection for {parameter}",
        labels={"anomaly_type": "Anomaly Type"},
    )
    fig.update_layout(
        legend_title_text="Anomaly Type",
        legend=dict(
            title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    fig.show()


@testModel.command()
def train(
    databasefilepath: Annotated[Path, typer.Argument(help="Path of the database file")],
    modelname: Annotated[
        str,
        typer.Argument(
            help=f"Name of the model to test, available models: {[name for name, obj in getmembers(train_model) if isfunction(obj) and obj.__module__ == 'train_model']}"
        ),
    ],
):
    """
    Train the model with name `modelname` to the database extracted from `databasefilepath`.

    Parameters:
    databasefilepath: path of the database
    modelname: string containing the function name for the model, for example isolationForest, available functions are in train_model.py
    """
    print("reading:", databasefilepath)
    resultsDatabaseFilePath =  databasefilepath.parents[1] / "testModelResults"/ f"{modelname}_anomaly_results.csv"
    print("save to:", resultsDatabaseFilePath)
    df = pd.read_csv(databasefilepath, sep=";")
    print(df.head())
    parameters = ["ETP", "GLOT", "RR", "TN", "TX"]
    df = preprocessing(df, parameters)
    print(df.head())
    model = getattr(train_model, modelname)
    model(df, parameters)
    # print(df)
    df.to_csv(resultsDatabaseFilePath, index=False)


@testModel.command()
def check(
    resultsdatabasefilepath: Annotated[
        Path, typer.Argument(help="Path of the database file with the results")
    ],
):
    """
    Checks the model results which are written as csv in `resultsdatabasefilepath`.

    Parameters:
    resultsdatabasefilepath: path of the results database
    """
    print("reading", resultsdatabasefilepath)
    df = pd.read_csv(resultsdatabasefilepath)
    print(df.head())
    parameters = ["ETP", "GLOT", "RR", "TN", "TX"]
    for param in parameters:
        plot_anomalies(param, df)


if __name__ == "__main__":
    testModel()
