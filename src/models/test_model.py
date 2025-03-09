from inspect import getmembers, isfunction
from pathlib import Path

import pandas as pd
import train_model
import typer
from typing import List, Optional
from typing_extensions import Annotated

from plotting import plot_anomalies, PlotMode, save_results_to_word
from preprocessing import join_spatial_info, preprocessing

testModel = typer.Typer()


@testModel.command()
def train(
    databasefilepath: Annotated[Path, typer.Argument(help="Path of the database file")],
    modelname: Annotated[
        str,
        typer.Argument(
            help=f"Name of the model to test, available models: {[name for name, obj in getmembers(train_model) if isfunction(obj) and obj.__module__ == 'train_model']}"
        ),
    ],
    checkafter: Annotated[
        bool, typer.Option(help="Diretly run check afterwards")
    ] = False,
    parameters: Annotated[
        Optional[List[str]],
        typer.Option(
            help="Parameters to predict, to change run --parameters P1 --parameters P2 ..."
        ),
    ] = ["ETP", "GLOT", "RR", "TN", "TX"],
    joinspatial: Annotated[
        bool, typer.Option(help="Join also the spatial information to the training")
    ] = True,
    savereport: Annotated[
        bool, typer.Option(help="Save report with results in data/reports")
    ] = False,
):
    """
    Train the model with name `modelname` to the database extracted from `databasefilepath`.

    Parameters:
    databasefilepath: path of the database
    modelname: string containing the function name for the model, for example isolationForest, available functions are in train_model.py
    """
    print("reading:", databasefilepath)
    resultsDatabaseFilePath = (
        databasefilepath.parents[1]
        / "testModelResults"
        / f"{modelname}_anomaly_results.csv"
    )
    print("save to:", resultsDatabaseFilePath)
    df = pd.read_csv(databasefilepath, sep=";")
    print(df.head())
    # Convert date column to datetime format
    df["datemesure"] = pd.to_datetime(df["datemesure"])
    # Check if the df has been already preprocessed
    if not all(f"{param}_anomaly" in df.columns for param in parameters):
        df = preprocessing(df, parameters)
        df.to_csv(
            databasefilepath.parents[1] / "preprocessed" / databasefilepath.name,
            index=False,
            sep=";",
        )
    if joinspatial:
        # Check if df has been already preprocessed with spatial data
        if not all(
            spatialcolumn in df.columns
            for spatialcolumn in ["Latitude", "Longitude", "Altitude", "cluster"]
        ):
            stationspath = databasefilepath.parents[1] / "raw" / "stationsmeteo.csv"
            df = join_spatial_info(df, stationspath)
            df.to_csv(
                databasefilepath.parents[1] / "preprocessed" / databasefilepath.name,
                index=False,
                sep=";",
            )
    print(df.head())
    print(df.info())
    model = getattr(train_model, modelname)
    model(df, parameters, joinspatial)
    # print(df)
    df.to_csv(resultsDatabaseFilePath, index=False)
    if checkafter:
        check(resultsDatabaseFilePath)
    if savereport:
        reportPath = (
            databasefilepath.parents[1] / "reports" / f"{modelname}_anomaly_results.doc"
        )
        imagesDir = databasefilepath.parents[1] / "reports" / "figures"
        save_results_to_word(df, parameters, filename=reportPath, imagesdir=imagesDir)


@testModel.command()
def check(
    resultsdatabasefilepath: Annotated[
        Path, typer.Argument(help="Path of the database file with the results")
    ],
    mode: Annotated[
        PlotMode,
        typer.Option("--mode", "-m", help="Plot mode: temporal, spatial, or both"),
    ] = PlotMode.TEMPORAL.value,
    savereport: Annotated[
        bool, typer.Option(help="Save report with results in data/reports, do not plot.")
    ] = False,
):
    """
    Checks the model results which are written as csv in `resultsdatabasefilepath`.

    Parameters:
    resultsdatabasefilepath: path of the results database
    """
    print("reading", resultsdatabasefilepath)
    df = pd.read_csv(resultsdatabasefilepath)
    print(df.head())
    # Take only the parameters that have been trained/tested
    parameters = [
        string.split("_")[0] for string in df.columns if "is_anomaly" in string
    ]
    if savereport:
        modelname = resultsdatabasefilepath.name.split("_")[0]
        reportPath = (
            resultsdatabasefilepath.parents[1] / "reports" / f"{modelname}_anomaly_results.doc"
        )
        imagesDir = resultsdatabasefilepath.parents[1] / "reports" / "figures"
        save_results_to_word(df, parameters, filename=reportPath, imagesdir=imagesDir)
    else:
        for param in parameters:
            plot_anomalies(param, df, mode=mode)


if __name__ == "__main__":
    testModel()
