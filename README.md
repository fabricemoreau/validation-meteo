Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

## Project Organization


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
	├── .decontainer       <- Parameters for VSCode to develop with dev containers
	├── .vscode            <- Parameters for VSCode: (extensions recommandations) make sure these parameters are valid for every machine
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── requirements-dev.txt <- hand-written file list of libraries used for developpements
    ├── requirements-scripts.txt <-hand-written fil list of libraries used in the script
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── meteo_models_clustering.py <- main script to call
    │   │   └── models_clustering     <- all models are here: one subfolder per model. example below
    │   │       └── my_model
    │   │           ├── preprocessing.py <- some models need additionnal preprocessing to run. called by meteo_models.py
    │   │           ├── train.py <- script for training model: called by meteo_models.py
    │   │           └── hyperparametre_search.py <- some models have this script to manually call to search best hyperparameters
    │   │
    │   ├── streamlit  <- Scripts for streamlit
    │   │   └── app.py

## How to configure your PC to develop with this repository with VSCode
Instructions [Here](vscode-dev-config.md)

## Model usage
Help:
```python
python src/models/meteo_models_clustering.py --help
```

To use a clustering model:
```python
# step 1: additionnal preprocessing
python src/models/meteo_models_clustering.py prepare isolationForest data/processed/meteo_pivot_cleaned_2010-2024_0.1.csv --no-joinspatial
# this command generate model_preprocessed file
# step 2: train
python src/models/meteo_models_clustering.py train model_preprocessed --no-joinspatial
# this command generate trained_file
# step 3: report
python src/models/meteo_models_clustering.py check trained_file --no-joinspatial
```

To use autoencoder model: 
- adapt constants in each script on src/models/autoencodeur
- run each script 

## Streamlit usage
```python
streamlit run src/streamlit/app.py
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
