# WSDM Cup 2022

---

This is the source code of the solution of the team PolimiRank for the WSDM Cup 2022.
The solution consists in a three steps model applied on (almost) all the combinations of source and target datasets.
1) Some among the most common recommendation algorithms are used to predict the scores for the required user-item couples on each dataset;
2) The scores of each dataset are ensembled using a simple non-linear combination and a set of common Gradient Boosting Decision Trees models (LightGBM, XGBoost, CatBoost);
3) All the scores and the ensembles of all the datasets are ensembled together to obtain the final prediction.

# Reproducibility

## Docker image

If the Docker image of the framework is available, it already trains the models for the two target markets 
using the same features and hyperparameters used for the top-scoring solution in the competition leaderboard,
and outputs the predictions.

## Running form scratch

This requires a few steps and quite a lot of time to re-run the hyperparameter optimization 
of the recommendation algorithms and the ensemble models. 

First, we suggest creating the environment with all the dependencies that are required to run the experiments.

Install [Anaconda](https://www.anaconda.com/products/individual) and be sure to have a C
compiler installed on your system, as the code includes some [Cython](https://cython.readthedocs.io/en/latest/index.html)
implementations (we suggest following the installation instructions on the website).

If these requirements are fulfilled, the following commands create a virtual conda environment called 'wsdmcup' 
with all the dependencies satisfied, and installs the framework in it.

```console
bash create_env.sh
conda activate wsdmcup
sh install.sh
```

### Data extraction

Extract the competition data in the 'dataset' directory, then run the following commands (remember to activate the virtual environment):

```console
python merge_datasets.py
python parser.py
```

### Recommenders hyperparameter tuning

The script 'create_baselines.py' runs the hyperparameter optimization, if necessary, and generates the
submissions for all the recommenders used for our solution and all the datasets.

```console
python create_baselines.py
```


### First level ensemble hyperparameter tuning

The script 'tune_ensemble_firstlevel.py' runs the hyperparameter optimization, if necessary, and generates the
ensemble with the specified GBDT model for all the datasets.

```console
python tune_ensemble_firstlevel.py -s -g xgboost
python tune_ensemble_firstlevel.py -s -g lgbm
python tune_ensemble_firstlevel.py -s -g catboost
```

To run the optimization of the non-linear combination of the scores predicted by the recommenders and output the ensemble scores:

```console
python create_ensemble.py #Runs the optimization
python create_ensemble_basic.py #Outputs the ensemble scores
```


### Second level ensemble hyperparameter tuning

The script 'tune_ensemble_endtoend.py' runs the hyperparameter optimization, if necessary, and predicts the
final ensemble scores with the specified GBDT model for all the datasets.
It also generates the zip file for the submission in the homonym folder.

```console
python tune_ensemble_endtoend.py -s 
```


