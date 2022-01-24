# WSDM Cup 2022

---

This is the source code of the solution of the team **PolimiRank** for the **WSDM Cup 2022**.

The solution consists in a multi-stage model applied on all the combinations of source and target markets.
All the different combinations of the source markets are merged with one or both the target markets in order to obtain the different datasets used for cross-domain recommendation. 
The items in common benefit from the interactions coming from all the markets included in the dataset, while the users are kept distinct from market to market.

In the first stage of the model, some among the most common collaborative recommendation algorithms, including item-based, user-based, graph-based, matrix factorization and deep learning models, are used to predict the scores for the required user-item couples on each dataset.
In the second stage, the scores for each dataset are separately ensembled using a simple non-linear combination, and a set of common Gradient Boosting Decision Trees models (LightGBM, XGBoost, CatBoost).
In the third (and final) stage, all the scores of first and second stages of all the datasets are ensembled together to obtain a unique final prediction.
In the last two stages, latent factors of users and items (obtained through an ALS model) and basic statistics about the datasets are included.

The competition provides train, validation and test data.
In the first stage, for the score prediction on validation all the recommenders are trained on train data for the specific target market, while for the prediction on test all the recommenders are trained on the union between train and validation data.
In second and third stages, instead, models are trained for the learning to rank task directly on validation data of a specific target market (models for different target markets have been trained and optimized separately), using 5-fold cross validation to avoid overfitting.


# Reproducibility

## Docker image

If the Docker image trains the models for the two target markets using the same features and 
hyperparameters used for the top-scoring solution in the competition leaderboard and outputs the predictions.
The following commands run the image in a container and outputs the solution zip in the current folder:

```console
docker create -it --name wsdmcup cesarebernardis/wsdmcup-2022:latest
docker start wsdmcup
docker exec -it wsdmcup /bin/bash
bash /train_and_submit.sh
exit
docker cp wsdmcup:/home/WSDMCup2022/submission/submission.zip ./
docker stop wsdmcup
```

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


