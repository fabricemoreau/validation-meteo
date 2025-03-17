from inspect import getmembers, isfunction
from pathlib import Path
from enum import Enum

import pandas as pd
import train_model
import typer
from typing import List, Optional
from typing_extensions import Annotated

from plotting import plot_anomalies, PlotMode, save_results_to_word
from preprocessing import join_spatial_info, preprocessing


class SUBPATHS(Enum):
    """
    To define all subpaths where file are stored
    """

    DATA_RAW = "raw"
    DATA_PREPROCESSED = "preprocessed"
    MODEL_RESULTS = "testModelResults"
    REPORTS = "reports"
    FIGURES = "figures"
    BOTH = "both"


meteo_models = typer.Typer()


@meteo_models.command()
def prepare(
    modelname: Annotated[
        str,
        typer.Argument(
            help=f"Name of the model to test, available models: {[name for name, obj in getmembers(train_model) if isfunction(obj) and obj.__module__ == 'train_model']}"
        ),
    ],
    databasefilepath: Annotated[
        Path, typer.Argument(help="Path of the database file")
    ] = "data/processed/meteo_pivot_cleaned_2010-2024.csv",
    parameters: Annotated[
        Optional[List[str]],
        typer.Option(
            help="Parameters to predict, to change run --parameters P1 --parameters P2 ..."
        ),
    ] = ["ETP", "GLOT", "RR", "TN", "TX"],
    joinspatial: Annotated[
        bool, typer.Option(help="Join also the spatial information to the training")
    ] = True,
    random_state: Annotated[
        int, typer.Option(help="Set the random_state for reproducibility")
    ] = None,
):
    """
    Prepare to train the model with name `modelname` to the database extracted from `databasefilepath`.
    A preprocessed version of the database is saved in the the folder 'preprocessed'

    Parameters:
    databasefilepath: path of the database
    modelname: string containing the function name for the model, for example isolationForest, available functions are in train_model.py
    """
    print("reading:", databasefilepath)
    df = pd.read_csv(databasefilepath, sep=";", parse_dates=["datemesure"])
    print(df.head())

    # Check if the df has been already preprocessed
    if not all(f"{param}_anomaly" in df.columns for param in parameters):
        df = preprocessing(df, parameters)
    if joinspatial:
        # Check if df has been already preprocessed with spatial data
        if not all(
            spatialcolumn in df.columns
            for spatialcolumn in ["Latitude", "Longitude", "Altitude", "cluster"]
        ):
            stationspath = (
                databasefilepath.parents[1]
                / SUBPATHS.DATA_RAW.value
                / "stationsmeteo.csv"
            )
            df = join_spatial_info(df, stationspath, random_state=random_state)
    preprocessedDatabaseFilePath = (
        databasefilepath.parents[1]
        / SUBPATHS.DATA_PREPROCESSED.value
        / f"meteo_pivot_{'-'.join(parameters)}_{joinspatial}.csv"
    )
    print("save to:", preprocessedDatabaseFilePath)
    df.to_csv(
        preprocessedDatabaseFilePath,
        index=False,
        sep=";",
    )


@meteo_models.command()
def train(
    modelname: Annotated[
        str,
        typer.Argument(
            help=f"Name of the model to test, available models: {[name for name, obj in getmembers(train_model) if isfunction(obj) and obj.__module__ == 'train_model']}"
        ),
    ],
    databasefilepath: Annotated[
        Path,
        typer.Argument(
            help="Path of the database file. If the model need special preprocessing, use the preprocessed database created with 'prepare' command"
        ),
    ] = "data/processed/meteo_pivot_cleaned_2010-2024.csv",
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
    random_state: Annotated[
        int, typer.Option(help="Set the random_state for reproducibility")
    ] = None,
):
    """
    Train the model with name `modelname` to the database extracted from `databasefilepath`.

    Parameters:
    databasefilepath: path of the database
    modelname: string containing the function name for the model, for example isolationForest, available functions are in train_model.py
    """
    resultsDatabaseFilePath = (
        databasefilepath.parents[1]
        / SUBPATHS.MODEL_RESULTS.value
        / f"{modelname}_{'-'.join(parameters)}_{joinspatial}_anomaly_results.csv"
    )

    print("reading:", databasefilepath)
    df = pd.read_csv(databasefilepath, sep=";", parse_dates=["datemesure"])
    print(df.head())

    # Check if the df has been already preprocessed
    if not all(f"{param}_anomaly" in df.columns for param in parameters):
        raise Exception(
            "Columns  '<param>_anomaly' missing in the database, please input a file built with command 'prepare'"
        )
    if joinspatial:
        # Check if df has been already preprocessed with spatial data
        if not all(
            spatialcolumn in df.columns
            for spatialcolumn in ["Latitude", "Longitude", "Altitude", "cluster"]
        ):
            raise Exception(
                "Spatial Columns missing in the database, please input a file built with command 'prepare'"
            )
    print(df.head())
    print(df.info())
    model = getattr(train_model, modelname)
    model(df, parameters, joinspatial, random_state=random_state)
    # print(df)
    print("save to:", resultsDatabaseFilePath)
    df.to_csv(resultsDatabaseFilePath, index=False)
    if checkafter:
        check(resultsDatabaseFilePath)
    if savereport:
        reportPath = (
            databasefilepath.parents[1]
            / SUBPATHS.REPORTS.value
            / f"{modelname}_{'-'.join(parameters)}_{joinspatial}_anomaly_results.doc"
        )
        imagesDir = (
            databasefilepath.parents[1]
            / SUBPATHS.REPORTS.value
            / SUBPATHS.FIGURES.value
        )
        save_results_to_word(df, parameters, filename=reportPath, imagesdir=imagesDir)


@meteo_models.command()
def check(
    resultsdatabasefilepath: Annotated[
        Path, typer.Argument(help="Path of the database file with the results")
    ],
    mode: Annotated[
        PlotMode,
        typer.Option("--mode", "-m", help="Plot mode: temporal, spatial, or both"),
    ] = PlotMode.TEMPORAL.value,
    savereport: Annotated[
        bool,
        typer.Option(help="Save report with results in data/reports, do not plot."),
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
            resultsdatabasefilepath.parents[1]
            / SUBPATHS.REPORTS.value
            / f"{modelname}_{'-'.join(parameters)}_anomaly_results.doc"
        )
        imagesDir = (
            resultsdatabasefilepath.parents[1]
            / SUBPATHS.REPORTS.value
            / SUBPATHS.FIGURES.value
        )
        save_results_to_word(df, parameters, filename=reportPath, imagesdir=imagesDir)
    else:
        for param in parameters:
            plot_anomalies(param, df, mode=mode)


if __name__ == "__main__":
    meteo_models()
