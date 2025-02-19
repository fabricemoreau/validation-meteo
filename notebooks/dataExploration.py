# %% [markdown]
#  # Data cleaning and preprocessing

# %%
import os
from pathlib import Path
from IPython.display import display
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# %%
# dataPath = Path("../..") / "validation-meteo-data" / "donneesmeteo_2010-2024" / "donneesmeteo_2010-2024.csv"
dataPath = Path("..") / "data" / "raw" / "donneesmeteo_2010-2024_500stations.csv"
print("read data from ", dataPath)
# dataPath = Path("../..") / "validation-meteo-data" / "donneesmeteo_2011-2015_completes" / "donneesmeteo_2011-2015_completes.csv"
meteodf = pd.read_csv(dataPath, sep=";")
display(meteodf.head())
# %%
print("Parameters:", meteodf.libellecourt.unique())
print("Shape:", meteodf.shape)
print("-----Info------\n")
display(meteodf.info())
print("-----NaNs Count-----")
display(meteodf.isna().sum())
# %%
meteodfToUse = meteodf[
    ["codearvalis", "libellecourt", "datemesure", "valeur", "valeurorigine"]
]  # For the classification model we only need these columns
meteodfToUse.head()
# %%
parametersToUse = ["ETP", "GLOT", "RR", "TN", "TX"]
meteodfToUse = meteodfToUse[
    meteodfToUse.libellecourt.isin(parametersToUse)
]  # Select interesting parameters
display(meteodfToUse.head())
print("Parameters:", meteodfToUse.libellecourt.unique())
print("-----NaN count --------")
display(meteodfToUse.isna().sum())
# %%
meteodfToUse.fillna(
    {"valeurorigine": meteodfToUse.valeur}, inplace=True
)  # NaNs in valeurorigine means that there has been no correction of the original value
print("-----NaN count --------")
display(meteodfToUse.isna().sum())
# %%
meteodfToUse.replace(
    to_replace={"valeurorigine": -999}, value=np.nan, inplace=True
)  # -999 can be considered as actual NaNs
print("-----NaN count --------")
display(meteodfToUse.isna().sum())
meteodfToUse.valeurorigine.min()
# %%
meteodfToUse.datemesure = pd.to_datetime(
    meteodfToUse.datemesure.apply(lambda value: value.split(" ")[0])
)  # all measurements date at 00:00:00 as hour so we can just drop it
meteodfToUse.info()
# %%
meteodfToUse["correction"] = (meteodfToUse.valeur != meteodfToUse.valeurorigine).astype(
    int
)  # values that have been corrected have valeurorigine different from valeur (threshold selection can be inserted here to avoid little corrections)
meteodfToUse.head()
# %%
meteodfToUse["difference"] = meteodfToUse.valeur - meteodfToUse.valeurorigine
meteodfToUse.head()
# %% [markdown]
# # Plotting

# %%
imagesDir = dataPath.parents[1] / "reports" / "figures"
if not os.path.exists(imagesDir):
    print("create image directory")
    os.makedirs(imagesDir)


def plotVarDateParameter(
    df,
    varToPlot,
    dateUnit,
    varToPlotName=None,
    normalize=False,
    saveToDir=None,
    parameters=None,
):
    if varToPlotName is None:
        varToPlotName = varToPlot
    if parameters is None:
        parameters = df.libellecourt.unique()
    fig = go.Figure()

    for parameter in parameters:
        data = df[df.libellecourt == parameter]
        if normalize:
            normalization = data[varToPlot].abs().max()
        else:
            normalization = 1
        fig.add_trace(
            go.Scatter(
                x=data.datemesure, y=data[varToPlot] / normalization, name=parameter
            )
        )
    title = varToPlotName + " per " + dateUnit
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=dict(text=dateUnit)),
        yaxis=dict(title=dict(text=varToPlotName)),
    )
    fig.show()
    if saveToDir is not None:
        fig.write_image(saveToDir / (title + ".png"))


def histVarDateParameter(df, varToPlot, dateUnit, varToPlotName=None, saveToDir=None):
    if varToPlotName is None:
        varToPlotName = varToPlot
    title = varToPlotName + " per " + dateUnit
    fig = px.bar(
        df,
        x="datemesure",
        y=varToPlot,
        color="libellecourt",
        barmode="group",
        title=title,
        labels={"datemesure": dateUnit, "correction": varToPlotName},
    )
    fig.show()
    if saveToDir is not None:
        fig.write_image(saveToDir / (title + ".png"))


# %%

correctiondfY = (
    meteodfToUse.groupby([meteodfToUse.datemesure.dt.year, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per year
display(correctiondfY.head())
plotVarDateParameter(
    correctiondfY,
    "correction",
    "year",
    varToPlotName="Corrections normalized",
    normalize=True,
    saveToDir=imagesDir,
)
histVarDateParameter(
    correctiondfY,
    "correction",
    "year",
    varToPlotName="Corrections",
    saveToDir=imagesDir,
)
correctiondfYNaNs = (
    meteodfToUse[meteodfToUse.valeurorigine.isna()]
    .groupby([meteodfToUse.datemesure.dt.year, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per year
display(correctiondfYNaNs.head())
plotVarDateParameter(
    correctiondfYNaNs,
    "correction",
    "year",
    varToPlotName="Corrections normalized (nan values only)",
    normalize=True,
    saveToDir=imagesDir,
)
histVarDateParameter(
    correctiondfYNaNs,
    "correction",
    "year",
    varToPlotName="Corrections (nan values only)",
    saveToDir=imagesDir,
)
correctiondfYNoNaNs = (
    meteodfToUse[~meteodfToUse.valeurorigine.isna()]
    .groupby([meteodfToUse.datemesure.dt.year, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per year
display(correctiondfYNoNaNs.head())
plotVarDateParameter(
    correctiondfYNoNaNs,
    "correction",
    "year",
    varToPlotName="Corrections normalized (no nans)",
    normalize=True,
    saveToDir=imagesDir,
)
histVarDateParameter(
    correctiondfYNoNaNs,
    "correction",
    "year",
    varToPlotName="Corrections (no nans)",
    saveToDir=imagesDir,
)

# %%
correctiondfM = (
    meteodfToUse.groupby([meteodfToUse.datemesure.dt.month, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per month
display(correctiondfM.head())
plotVarDateParameter(correctiondfM, "correction", "month", "Corrections")
plotVarDateParameter(
    correctiondfM, "correction", "month", "Corrections normalized", normalize=True
)
histVarDateParameter(
    correctiondfM, "correction", "month", "Corrections", saveToDir=imagesDir
)
correctiondfMNaNs = (
    meteodfToUse[meteodfToUse.valeurorigine.isna()]
    .groupby([meteodfToUse.datemesure.dt.month, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per month
display(correctiondfMNaNs.head())
plotVarDateParameter(
    correctiondfMNaNs, "correction", "month", "Corrections (nan values only)"
)
plotVarDateParameter(
    correctiondfMNaNs,
    "correction",
    "month",
    "Corrections normalized (nan values only)",
    normalize=True,
)
histVarDateParameter(
    correctiondfMNaNs,
    "correction",
    "month",
    "Corrections (nan values only)",
    saveToDir=imagesDir,
)
correctiondfMNoNaNs = (
    meteodfToUse[~meteodfToUse.valeurorigine.isna()]
    .groupby([meteodfToUse.datemesure.dt.month, meteodfToUse.libellecourt])
    .correction.sum()
    .reset_index()
)  # Number of corrections per parameter per month
display(correctiondfMNoNaNs.head())
plotVarDateParameter(
    correctiondfMNoNaNs, "correction", "month", "Corrections (no nans)"
)
plotVarDateParameter(
    correctiondfMNoNaNs,
    "correction",
    "month",
    "Corrections normalized (no nans)",
    normalize=True,
)
histVarDateParameter(
    correctiondfMNoNaNs,
    "correction",
    "month",
    "Corrections (no nans)",
    saveToDir=imagesDir,
)
# %%
valeurdfM = (
    meteodfToUse.groupby([meteodfToUse.datemesure.dt.month, meteodfToUse.libellecourt])
    .valeur.mean()
    .reset_index()
)  # Valeur vs month per parameter
display(valeurdfM.head())
plotVarDateParameter(
    valeurdfM,
    "valeur",
    "month",
    varToPlotName="valeur normalized",
    normalize=True,
    saveToDir=imagesDir,
)
plotVarDateParameter(valeurdfM, "valeur", "month", varToPlotName="valeur")
histVarDateParameter(valeurdfM, "valeur", "month", "valeur")
# %%
valeurdfY = (
    meteodfToUse.groupby([meteodfToUse.datemesure.dt.year, meteodfToUse.libellecourt])
    .valeur.mean()
    .reset_index()
)
display(valeurdfY.head())
plotVarDateParameter(
    valeurdfY,
    "valeur",
    "year",
    varToPlotName="valeur normalized",
    normalize=True,
    saveToDir=imagesDir,
)
plotVarDateParameter(
    valeurdfY,
    "valeur",
    "year",
    varToPlotName="temperature normalized",
    normalize=True,
    parameters=["TX", "TN"],
    saveToDir=imagesDir,
)
plotVarDateParameter(valeurdfY, "valeur", "year", varToPlotName="valeur")
histVarDateParameter(valeurdfY, "valeur", "year")
# %%
correctedDF = meteodfToUse[meteodfToUse.correction == 1]
display(correctedDF.head())
differencedfY = (
    correctedDF.groupby([correctedDF.datemesure.dt.year, correctedDF.libellecourt])
    .difference.mean()
    .reset_index()
)
valOrigCorrectedDfY = (
    correctedDF.groupby([correctedDF.datemesure.dt.year, correctedDF.libellecourt])
    .valeurorigine.mean()
    .reset_index()
)
differencedfY = differencedfY.merge(valOrigCorrectedDfY)
differencedfY["relative_corr"] = differencedfY.difference / differencedfY.valeurorigine
display(differencedfY.head())
plotVarDateParameter(
    differencedfY, "difference", "year", "valeur - valeurorigine", normalize=True
)
plotVarDateParameter(
    differencedfY, "relative_corr", "year", "(valeur - valeurorigine)/valeurorigine"
)
histVarDateParameter(
    differencedfY, "relative_corr", "year", "relative correction", saveToDir=imagesDir
)

# %%
differencedfM = (
    correctedDF.groupby([correctedDF.datemesure.dt.month, correctedDF.libellecourt])
    .difference.mean()
    .reset_index()
)
valOrigCorrectedDfM = (
    correctedDF.groupby([correctedDF.datemesure.dt.month, correctedDF.libellecourt])
    .valeurorigine.mean()
    .reset_index()
)
differencedfM = differencedfM.merge(valOrigCorrectedDfM)
differencedfM["relative_corr"] = differencedfM.difference / differencedfM.valeurorigine
display(differencedfM.head())
plotVarDateParameter(
    differencedfM,
    "difference",
    "month",
    "valeur - valeurorigine normalized",
    normalize=True,
)
plotVarDateParameter(
    differencedfM, "relative_corr", "month", "(valeur - valeurorigine)/valeurorigine"
)
histVarDateParameter(
    differencedfM, "relative_corr", "month", "relative correction", saveToDir=imagesDir
)
histVarDateParameter(
    differencedfM[differencedfM.libellecourt != "TN"],
    "relative_corr",
    "month",
    "relative correction (no TN)",
    saveToDir=imagesDir,
)

# %%
meteodfToUse["year"] = meteodfToUse.datemesure.dt.year
yearsPerStation = meteodfToUse.groupby(["codearvalis"]).year.nunique().reset_index()
title = "Distribution of the number of active years per stations"
fig = px.histogram(
    yearsPerStation, x="year", labels={"year": "number of years active"}, title=title
)
fig.update_layout(bargap=0.2)
fig.show()
fig.write_image(imagesDir / (title + ".png"))

# %%
stationsPerYear = (
    meteodfToUse.groupby([meteodfToUse.datemesure.dt.year])
    .codearvalis.nunique()
    .reset_index()
)
title = "Number of active stations per year"
fig = px.bar(
    stationsPerYear,
    x="datemesure",
    y="codearvalis",
    labels={"codearvalis": "number of active stations", "datemesure": "year"},
    title=title,
)
fig.show()
fig.write_image(imagesDir / (title + ".png"))

# %%
