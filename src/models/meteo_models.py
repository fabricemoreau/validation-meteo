from pkgutil import iter_modules
from pathlib import Path
from enum import Enum

import os
import pandas as pd
import models as models
import typer
from typing import List, Optional
from typing_extensions import Annotated

from plotting import plot_anomalies, PlotMode, save_results_to_word

# list of models
list_models = []
for importer, modname, ispkg in iter_modules(models.__path__):
    if ispkg:
        list_models.append(modname)


class SUBPATHS(Enum):
    """
    To define all subpaths where file are stored
    """

    DATA_RAW = "raw"
    DATA_PREPROCESSED = "preprocessed"
    MODEL_RESULTS = "testModelResults"
    REPORTS = "reports"
    FIGURES = "figures"


meteo_models = typer.Typer()


@meteo_models.command()
def prepare(
    modelname: Annotated[
        str,
        typer.Argument(
            help=f"Name of the model to test, available models: {list_models}"
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
    print("preparing folders")
    for subpath in SUBPATHS:
        # SUBPATHS.FIGURES is a special case
        if subpath != SUBPATHS.FIGURES:
            path = databasefilepath.parents[1] / subpath.value
            if not os.path.exists(path):
                print(f"create {path} directory")
                os.makedirs(path)
        else:
            path = (
                databasefilepath.parents[1]
                / SUBPATHS.REPORTS.value
                / SUBPATHS.FIGURES.value
            )
            if not os.path.exists(path):
                print(f"create {path} directory")
                os.makedirs(path)

    # import module
    try:
        print(f"Lazy loading {modelname} train module")
        preprocessing_module = __import__(
            f"models.{modelname}.preprocessing", fromlist="preprocessing"
        )
    except ModuleNotFoundError:
        print("No preprocessing for {modelname}")
    else:
        print("reading:", databasefilepath)
        df = pd.read_csv(databasefilepath, sep=";", parse_dates=["datemesure"])
        print(df.head())

        stationspath = (
            databasefilepath.parents[1] / SUBPATHS.DATA_RAW.value / "stationsmeteo.csv"
        )

        df = preprocessing_module.preprocessing(
            df, stationspath, parameters, joinspatial, random_state
        )

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
            help=f"Name of the model to test, available models: {list_models}"
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
    log: Annotated[
        bool,
        typer.Option(help="Save training to log file in data/testModelResults"),
    ] = True,
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
    # import module
    print(f"Lazy loading {modelname} train module")
    train_module = __import__(f"models.{modelname}.train", fromlist="train")

    resultsDatabaseFilePath = (
        databasefilepath.parents[1]
        / SUBPATHS.MODEL_RESULTS.value
        / f"{modelname}_{'-'.join(parameters)}_{joinspatial}_anomaly_results.csv"
    )
    if log:
        logFilePath = (
            databasefilepath.parents[1]
            / SUBPATHS.MODEL_RESULTS.value
            / f"{modelname}_{'-'.join(parameters)}_{joinspatial}_log.txt"
        )
    else:
        logFilePath = None

    print("reading:", databasefilepath)
    df = pd.read_csv(databasefilepath, sep=";", parse_dates=["datemesure"])
    print(df.head())
    print(df.info())

    df = train_module.train(
        df, parameters, joinspatial, log_file=logFilePath, random_state=random_state
    )
    # keep only test data
    df = df[df.is_test == 1]
    # print(df)
    print("save to:", resultsDatabaseFilePath)
    df.to_csv(resultsDatabaseFilePath, index=False)
    if checkafter:
        check(resultsDatabaseFilePath, savereport=savereport)


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
    df = pd.read_csv(resultsdatabasefilepath, parse_dates=["datemesure"])
    print(df.head())
    # Take only the parameters that have been trained/tested
    parameters = [
        string.split("_")[0] for string in df.columns if "_anomaly_pred" in string
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
        save_results_to_word(
            df, parameters, modelname, filename=reportPath, imagesdir=imagesDir
        )
    else:
        for param in parameters:
            plot_anomalies(param, df, mode=mode)


if __name__ == "__main__":
    meteo_models()
