import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
import pytensor
import xarray as xr

from datetime import datetime, timedelta
import os
import glob
import yaml
import joblib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed

from visualizations import *
from validations import discrete_evaluations, check_feature_importance, tune_prob_threshold
from reports import *
from utils import *

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE #I dont usually prefer this but trying it out

#Classifiers
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

#NN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Configs
#os.chdir("/Users/echhitjoshi/Library/Mobile Documents/com~apple~CloudDocs/Work/EonJive")
if os.uname().machine.lower() == "arm64":
    pytensor.config.cxx = '/usr/bin/clang++'

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
home_dir = config['HOME_DIRECTORY']


def plot_trace(trace):
    pm.plot_trace(trace,figsize = (15,8),legend = True)
    plt.show()


def load_latest_model(directory = home_dir + '/models', pattern = '*'):
    files = glob.glob(os.path.join(directory,pattern))
    print("Reading from ",files)
    if not files:
        return None
    latest_file = max(files,key = os.path.getmtime)
    print('latest model is : ', latest_file)
    trace = az.from_netcdf(latest_file)
    return trace

def load_last_model(filepath = home_dir + '/models/'):
    models = os.listdir(filepath)
    models.sort(reverse = True)
    trace = az.from_netcdf(filepath + models[0])
    return trace
    
def get_params(algorithm_name,pipeline_key):
    '''algorithm_name: select from [lgbm,xgb,catboost,rf]'''
    if 'lgbm' in algorithm_name:
        return {
            pipeline_key+'__n_estimators': [50,100, 300, 500],
            pipeline_key+'__learning_rate': [0.01, 0.05, 0.1],
            pipeline_key+'__max_depth': [3, 5, 7, 10],
            pipeline_key+'__num_leaves': [15, 31, 63],
            pipeline_key+'__min_child_samples': [10, 20, 50],
            pipeline_key+'__colsample_bytree': [0.6, 0.8, 1.0],
            pipeline_key+'__subsample': [0.6, 0.8, 1.0],
            pipeline_key + '__is_unbalance': [True, False]
            
        }
    elif 'catboost' in algorithm_name:
        return  {
        pipeline_key+'__iterations': [100, 300, 500],
        pipeline_key+'__depth': [4, 6, 8, 10,15],
        pipeline_key+'__learning_rate': [0.01, 0.05, 0.1],
        pipeline_key+'__l2_leaf_reg': [1, 3, 5, 7],
        pipeline_key+'__bagging_temperature': [0, 0.5, 1, 2],
        pipeline_key+'__border_count': [32, 64, 128],
        pipeline_key+'__class_weights': [[1, 1], [1, 3], [1, 5]],
    }
    elif 'xgb' in algorithm_name:
        return {
            pipeline_key+'__n_estimators': [50,100, 300, 500],
            pipeline_key+'__learning_rate': [0.01, 0.05, 0.1],
            pipeline_key+'__max_depth': [3, 5, 7, 10],
            pipeline_key+'__gamma': [0,1,5,7],
            pipeline_key+'__min_child_weight': [1,5,10,15],
            pipeline_key+'__subsample': [0.6, 0.8, 1.0],
            pipeline_key+'__colsample_bytree': [0.6, 0.8, 1.0],
            pipeline_key+'__scale_pos_weight': [1, 3, 5],
        }
    elif 'rf' in algorithm_name:
        return {
            pipeline_key + '__n_estimators': [100, 200, 500],
            pipeline_key + '__max_depth': [None, 5, 10, 20, 30],
            pipeline_key + '__min_samples_split': [2, 5, 10],
            pipeline_key + '__min_samples_leaf': [1, 2, 4],
            pipeline_key + '__max_features': ['sqrt', 'log2', None],
            pipeline_key + '__bootstrap': [True, False],
            pipeline_key + '__criterion': ['gini', 'entropy', 'log_loss'],
            pipeline_key + '__class_weight': [None, 'balanced'],
            pipeline_key + '__class_weight': ['balanced', 'balanced_subsample', None, {0: 1, 1: 5}],
        }
    

# Models with feature selection and gridsearch   

def run_model_with_fs_tune(train_X,test_X,train_y,test_y,dat_dict,algorithm,output_path):

    # Create Copies to work within the function
    train_X = train_X.copy()
    test_X = test_X.copy()

    categorical_features = dat_dict[(dat_dict['type'] == 'categorical') & (dat_dict['modeling_feature'] == 1)]['feature'].values
    categorical_features_indices = train_X.columns.get_indexer(categorical_features)
    categorical_features_indices = categorical_features_indices[categorical_features_indices !=-1]
    print(categorical_features_indices)

    
    if algorithm == 'catboost':
        #Categorical features
        print("algorithm used is catboost")
        categorical_features = dat_dict[(dat_dict['type'] == 'categorical') & (dat_dict['modeling_feature'] == 1)]['feature'].values
        categorical_features_indices = train_X.columns.get_indexer(categorical_features)
        categorical_features_indices = categorical_features_indices[categorical_features_indices !=-1]
        print(categorical_features_indices)
        # handle nulls here:
        for col_idx in categorical_features_indices:
            col_name = train_X.columns[col_idx]
            train_X[col_name] = train_X[col_name].replace(np.nan, 'NaN_string').astype(str) # Convert NaN to 'NaN_string' and then to string
            test_X[col_name] = test_X[col_name].replace(np.nan, 'NaN_string').astype(str) # Do the same for test data

        # train_pool = Pool(data = train_X, label = train_y, cat_features = categorical_features_indices)

        # test_pool = Pool(data = test_X, label = test_y, cat_features = categorical_features_indices)

        cb = CatBoostClassifier(
                                loss_function='Logloss', 
                                eval_metric='AUC', 
                                random_seed=42, 
                                verbose=10,
                                class_weights=[1,2])  #Testing Class weights
        

        param_grid = {
            'iterations': [100, 300, 500],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'bagging_temperature': [0, 0.5, 1, 2],
            'border_count': [32, 64, 128]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True , random_state=42)

        # Search
        search = RandomizedSearchCV(
            estimator=cb,
            param_distributions=param_grid,
            scoring='roc_auc',
            n_iter=30,
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        search.fit(train_X,train_y,cat_features = categorical_features_indices,early_stopping_rounds = 150, eval_set = (test_X,test_y))
        
    else:
        # Ordinal encoder
        ordinal_encodings = dict()
        categorical_features = dat_dict[(dat_dict['type'] == 'categorical') & (dat_dict['modeling_feature'] == 1)]['feature'].values
        for col in train_X.columns:
            
            if col in categorical_features:
                print(f"encoding feature {col}")
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= -1)
                train_X[col] = oe.fit_transform(train_X[[col]])
                test_X[col] = oe.transform(test_X[[col]])
                #save encoding object for deployment:
                ordinal_encodings[col] = oe
                # save to disc in submodel path
                joblib.dump(oe,output_path+f'/encodings/{col}_categorical_encodings.pkl')
    
    
            # Use rf for feature selection
            feature_selector = SelectFromModel( estimator= RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=33,
                
            ),threshold = 'median')
            
            # Initialize Classifier
            if algorithm == 'lgbm':
                final_model = LGBMClassifier(random_state=33, n_jobs=6)
            elif algorithm == 'xgb':
                final_model = XGBClassifier(random_state=33, n_jobs=6)
            elif algorithm == 'rf':
                final_model = RandomForestClassifier(random_state=33,n_jobs=6)

            # Param Grid
            classifier_model_pipeline_key = 'classifier_model'
            param_grid = get_params(algorithm,classifier_model_pipeline_key)

            # Create pipeline
            pipe = Pipeline([
                ('feature_selection', feature_selector),
                (classifier_model_pipeline_key, final_model)
            ])

            # GridSearch parameter space
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                scoring='roc_auc',
                n_iter=50,
                cv=cv,
                verbose=1,
                random_state=33,
                n_jobs=-1
            )

        # Fit tuned models
        search.fit(train_X, train_y)
    
    best_model = search.best_estimator_
    pred = best_model.predict(test_X)
    pred_proba = best_model.predict_proba(test_X)[:, 1]
    
    if algorithm == 'catboost':
        best_model.fit(train_X,train_y,cat_features = categorical_features_indices, eval_set = (test_X,test_y))
    else:
        best_model.fit(train_X,train_y)
    

    # plot feature importance:
    if algorithm != 'catboost':
        importances = best_model.named_steps['classifier_model'].feature_importances_
        mask = best_model.named_steps['feature_selection'].get_support()
        selected_features = train_X.columns[mask]
        feat_imp_df = pd.DataFrame({
            'feature_names': selected_features,
            'feature_importance': importances
        }).sort_values(by='feature_importance', ascending=False)
    else:
        importances = best_model.get_feature_importance()
        feature_names = train_X.columns
        feat_imp_df = pd.DataFrame({
            'feature_names': feature_names,
            'feature_importance': importances
        }).sort_values(by='feature_importance', ascending=False)
    plot_feature_importance(feat_imp_df)

    # Training Metrics
    train_pred = best_model.predict(train_X)
    train_pred_proba = best_model.predict_proba(train_X)[:,1]
    print("Training Metrics: \n")
    discrete_evaluations(train_y,train_pred,train_pred_proba,'train',classification_type="binomial",model_path=output_path)

    # Test Metrics
    pred = best_model.predict(test_X)
    pred_proba = best_model.predict_proba(test_X)[:,1]
    print(pred_proba)

    discrete_evaluations(test_y,pred,pred_proba,'test',classification_type="binomial",model_path= output_path)

    thres_new = tune_prob_threshold(test_y,pred_proba)
    pred_new = np.where(pred_proba > thres_new['tpr_fpr'] - 0.03,1,0) # Less strict 


    thres_df = pd.DataFrame(list(thres_new.items()), columns=['threshold', 'value'])
    pd.DataFrame(thres_df).to_csv(output_path+"/thresholds.csv",index = False)

    print("\n Evaluations after probability threshold tuning: ")
    discrete_evaluations(test_y,pred_new,pred_proba,'test_parameter_tuned',classification_type="binomial",model_path=output_path)

    dat_dict.to_csv(output_path+'/dat_dict.csv')
    joblib.dump(best_model, output_path+ '/' + algorithm +'.pkl')

    return best_model


def predict_final(model_path, test):
    #Make Copy of test
    test = test.copy()

    dat_dict = pd.read_csv(model_path+'/dat_dict.csv')
    dat_dict.drop(columns =['Unnamed: 0'],inplace =True)
    dat_dict
    
    # Load model
    model_file = glob.glob(model_path + '/*.pkl')
    if not model_file:
        raise FileNotFoundError("No model .pkl file found in given path.")
    model = joblib.load(model_file[0])

    
    # Check Encodings
    if 'catboost' in model_path or 'logistic' in model_path:
        print("No encodings used.")
    else:
        print("Loading encodings...")
        encoding_path = Path(model_path) / "encodings"
        encoding_files = list(encoding_path.glob('*.pkl'))
        if not encoding_files:
            print("No encoding files found. Skipping encoding.")
        else:
            for file in encoding_files:
                parts = file.name.split("_cat")
                if len(parts) < 2:
                    print(f"Skipping malformed encoding file: {file.name}")
                    continue
                col = parts[0]
                if col in test.columns:
                    print(f"Encoding column: {col}")
                    encoder = joblib.load(file)
                    test[col] = test[col].astype(str)
                    test[col] = encoder.transform(test[[col]])
                else:
                    print(f"Column '{col}' not found in test set. Skipping.")

    # Process data accordingly
    if 'catboost' not in model_path:
        print("Applying pipeline preprocessing")
        selector = model.named_steps['feature_selection']
        feature_names_before_selection = selector.feature_names_in_
        selected_mask = selector.get_support()
        selected_features = feature_names_before_selection[selected_mask]

        # Ensure all expected columns are present in test
        for col in feature_names_before_selection:
            if col not in test.columns:
                print(f"Adding missing column: {col}")
                test[col] = 0

        # Reorder test columns to match training
        X_new = test[feature_names_before_selection]
        #X_new = test[selected_features] # if model takes only selected features as input

    else:
        print("Applying CatBoost preprocessing")
        categorical_features = dat_dict[
            (dat_dict['type'] == 'categorical') &
            (dat_dict['modeling_feature'] == 1)
        ]['feature'].values

        for col in categorical_features:
            if col in test.columns:
                test[col] = test[col].replace(pd.NA, 'NaN_string').fillna('NaN_string').astype(str)
            else:
                print(f"Adding missing categorical column: {col}")
                test[col] = 'NaN_string'

        X_new = test[model.feature_names_]

    print("Input columns used for prediction:")
    print(X_new.columns.tolist())

    print("Generating predictions")

    print(X_new)
    pred_proba = model.predict_proba(X_new)[:, 1]

    # Load threshold from file
    threshold_file = Path(model_path) / "thresholds.csv"
    if threshold_file.exists():
        thres_df = pd.read_csv(threshold_file)
        threshold_row = thres_df[thres_df['threshold'] == 'tpr_fpr']
        if not threshold_row.empty:
            threshold_value = float(threshold_row['value'].values[0])
        else:
            print("Threshold 'tpr_fpr' not found, defaulting to 0.5.")
            threshold_value = 0.5
    else:
        print("Thresholds.csv not found. Defaulting to 0.5.")
        threshold_value = 0.5

    preds = np.where(pred_proba >= threshold_value - 0.02, 1, 0)

    # Final Data
    final_pred = pd.DataFrame({
        'class': preds,
        'pred_probability': pred_proba
    })

    print("Prediction Class Distribution")
    print(final_pred['class'].value_counts(normalize = True))

    return final_pred


def outcome_prediction():
    pass


def check_elbow(n_clusters,dat):
    km = KMeans(n_clusters,random_state=33)
    km.fit(dat)
    return km


def fit_kmeans(dat,target:list,k = None,cluster_colname = 'cluster',model_path = None):
    '''
    dat: data with target col
    target: column name
    '''
    dat = dat.copy() # Work on a copy

    # Create kmeans directory:
    if not model_path:
        model_path = get_latest_model_folder(home_dir + "/outputs/models")
        
    kmeans_dir = os.path.join(model_path, "kmeans")
    if not os.path.exists(kmeans_dir):
        os.makedirs(kmeans_dir)

    # --- IMPORTANT: Sort feature names to ensure consistent order ---
    target = sorted(target)

    if not k:
        k_range = range(1,25)

        with ProcessPoolExecutor(max_workers=5) as executor:
            # Use dat[target].dropna() for elbow check as well
            futures = {executor.submit(check_elbow, i, dat[target].dropna(axis=0)): i for i in k_range}

            results = {}
            for future in as_completed(futures):
                k_val = futures[future]
                try:
                    kmeans_val = future.result()
                    results[k_val] = kmeans_val.inertia_
                except Exception as e:
                    print(f"Error checking elbow for k={k_val}: {e}")

        ks = sorted(results.keys())
        inertias = [results[k] for k in ks]
        
        if len(inertias) > 2:
            first_deriv = np.diff(inertias)/np.diff(ks)
            second_deriv = np.diff(first_deriv)
            elbow_idx = np.argmin(np.abs(second_deriv)) + 1
            elbow_k = ks[elbow_idx]
            print(f"Elbow method suggests k = {elbow_k}")
            k = elbow_k
        else:
            print("Not enough data points to determine elbow, defaulting to k=10")
            k = 10

        # Plotting
        plt.plot(ks, inertias, marker='o')
        if 'elbow_k' in locals():
             plt.axvline(x = elbow_k, color = 'r', linestyle = '--', label = f'Elbow at k = {elbow_k}')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Curve')
        plt.legend()
        plt.show()


    final_kmeans = KMeans(n_clusters=k, random_state=33, n_init='auto')
    
    dat_for_clustering = dat[target].dropna()
    print(f"Dropping {dat.shape[0] - dat_for_clustering.shape[0]} rows due to nulls!")
    
    labels = final_kmeans.fit_predict(dat_for_clustering)
    dat.loc[dat_for_clustering.index, f"{cluster_colname}"] = labels

    if 'win' in dat.columns:
        cluster_win = dat.groupby(['cluster'],as_index = False).agg(games = ('cluster','size'),win_perc = ('win','mean')).sort_values('win_perc', ascending = False)
        cluster_win['cluster_rank'] = cluster_win['win_perc'].rank(ascending = False).astype('int')
        dat = dat.merge(cluster_win,how = 'left',on = 'cluster')

    # --- Save the model and the feature list ---
    joblib.dump(final_kmeans, os.path.join(kmeans_dir, "kmeans_model.pkl"))
    joblib.dump(target, os.path.join(kmeans_dir, "kmeans_features.pkl"))

    centers = final_kmeans.cluster_centers_
    col_names = dat_for_clustering.columns
    df_centers = pd.DataFrame(centers, columns=col_names)
    
    # --- Save cluster centers at the root of the model path ---
    df_centers.to_csv(os.path.join(model_path, "kmeanscluster_centers.csv"), index=False)


    from sklearn.decomposition import PCA
    # Reduce dimensions for plotting
    X_reduced = PCA(n_components=2).fit_transform(dat_for_clustering)

    # Reduce cluster centers into PCA space as well
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(dat_for_clustering)
    centers_reduced = pca.transform(final_kmeans.cluster_centers_)

    # Cluster plots
    plt.figure(figsize=(8,6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                c='black', marker='X', s=200, label='Centers')
    plt.title("KMeans clusters in PCA space")
    plt.legend()
    plt.show()




    return dat

def predict_kmeans(dat,model_path):
    """
    dat: pd.DataFrame to predict on 
    model_path : path to model with kmeans folder
    """
    if os.path.exists(model_path):
        kmeans_model_path = os.path.join(model_path,"kmeans","kmeans_model.pkl")
        kmeans_features_path = os.path.join(model_path,"kmeans","kmeans_features.pkl")
        
        
        
        if not os.path.exists(kmeans_features_path):
                # Fallback for older models that don't have saved features
                print("Warning: kmeans_features.pkl not found. Using a hardcoded list of features. This may lead to errors if the model was trained with a different feature set.")
                kmeans_cols = sorted(['games_rating', 'shots_total', 'shots_on', 'passes_total', 'passes_key', 'passes_accurate', 'duels_total', 'duels_won', 'fouls_drawn', 'cards_yellow', 'tackles_interceptions', 'tackles_blocks', 'dribble_success_rate', 'dribbles_past', 'target_shot_conversion_perc', 'duels_won_perc', 'pass_accuracy_perc', 'fouls_committed', 'penalty_won', 'penalty_commited'])
        else:
                kmeans_cols = joblib.load(kmeans_features_path)

        kmeans_model = joblib.load(kmeans_model_path)

        dat.dropna(subset=kmeans_cols, inplace=True)
        
        # Ensure columns are in the correct order
        kmeans_dat = dat[kmeans_cols]
        
        labels = kmeans_model.predict(kmeans_dat)
        dat['cluster'] = labels

        cluster_win = dat.groupby('cluster')['win'].mean().reset_index(name='win_perc')
        cluster_win['cluster_rank'] = cluster_win['win_perc'].rank(ascending=False).astype(int)
        
        dat = dat.merge(cluster_win[['cluster', 'cluster_rank']], on='cluster')    
        
    return dat

def train_kmeans_and_get_clusters(player_stats_df, n_clusters=50):
    """
    Trains a KMeans model for each season on player stats and 
    returns the dataframe with cluster labels.
    """
    features_for_clustering = [col for col in player_stats_df.columns if '_per_90_percentile' in col]
    
    # Ensure cluster column exists
    if 'cluster' not in player_stats_df.columns:
        player_stats_df['cluster'] = -1 # default value

    for season in player_stats_df['season'].unique():
        season_data_indices = player_stats_df[player_stats_df['season'] == season].index
        
        if len(season_data_indices) == 0:
            continue

        clustering_data = player_stats_df.loc[season_data_indices, features_for_clustering].fillna(0)
        
        if clustering_data.empty:
            continue

        # Adjust n_clusters if less samples than clusters
        k = min(n_clusters, len(clustering_data))
        if k <= 1: # Not enough samples to cluster
            player_stats_df.loc[season_data_indices, 'cluster'] = -1
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(clustering_data)
        
        player_stats_df.loc[season_data_indices, 'cluster'] = clusters

    return player_stats_df


def bayesian_team_ability_model(data, team_col, opponent_col, season_col, outcome_col, games_played_col, output_path=None, save_strategy='summary', thinning_factor=10):
    """
    This function creates and samples from a Bayesian hierarchical model to estimate team ability.
    It saves the model summary or a thinned trace and mappings for later use.

    Args:
        data (pd.DataFrame): The input data.
        team_col (str): The name of the column containing team names.
        opponent_col (str): The name of the column containing opponent names.
        season_col (str): The name of the column containing season information.
        outcome_col (str): The name of the column containing the outcome of the game (wins).
        games_played_col (str): The name of the column containing the number of games played.
        output_path (str, optional): Path to save the model artifacts and mappings. Defaults to None.
        save_strategy (str, optional): The strategy for saving model artifacts. One of ['thin', 'summary']. Defaults to 'summary'.
        thinning_factor (int, optional): The factor by which to thin the trace if save_strategy is 'thin'. Defaults to 10.

    Returns:
        trace: The trace of the PyMC model.
    """
    if data.empty:
        raise ValueError("Input data is empty. Cannot build the Bayesian model.")

    # Clean data before processing
    required_cols = [team_col, opponent_col, season_col, outcome_col, games_played_col]
    data = data.dropna(subset=required_cols).copy()

    if data.empty:
        raise ValueError("Input data is empty after dropping rows with missing values in required columns.")

    # Ensure correct dtypes for binomial likelihood
    data[outcome_col] = data[outcome_col].astype(int)
    data[games_played_col] = data[games_played_col].astype(int)

    season_v = data[season_col]
    team_v = data[team_col].astype(str)
    opps_v = data[opponent_col].astype(str)

    # Consolidate all teams from both columns
    all_teams = np.union1d(team_v.unique(), opps_v.unique())

    season_map = {v: i for i, v in enumerate(season_v.unique())}
    team_map = {v: i for i, v in enumerate(all_teams)}

    season_ix = season_v.map(season_map).values
    team_ix = team_v.map(team_map).values
    opps_ix = opps_v.map(team_map).values

    coords = {
        "season": list(season_map.keys()),
        "team": list(team_map.keys()),
        "opps": list(team_map.keys())  # Use all teams for opponents as well
    }

    with pm.Model(coords=coords) as model:
        # team prior
        mu_team = pm.Normal("mu_team", 0, 1)
        sigma_team = pm.HalfNormal("sigma_team", 2)

        # opponent prior
        mu_opps = pm.Normal("mu_opps", 0, 2)
        sigma_opps = pm.HalfNormal("sigma_opps", 3)

        # Team ability (not opponent-specific)
        theta_raw = pm.Normal("theta_raw", 0, 1, dims=("season", "team"))
        theta_t = mu_team + theta_raw * sigma_team
        theta = pm.Deterministic("theta", theta_t - theta_t.mean(axis=1, keepdims=True), dims=("season", "team"))

        beta_raw = pm.Normal("beta_raw", 0, 1, dims=("season", "opps"))
        beta_t = mu_opps + beta_raw * sigma_opps
        beta = pm.Deterministic("beta", beta_t - beta_t.mean(axis=1, keepdims=True), dims=("season", "opps"))

        # Logit: team ability - opps ability/difficulty
        logit = theta[season_ix, team_ix] - beta[season_ix, opps_ix]
            
        p = pm.Deterministic("p", pm.math.sigmoid(logit))
        n = data[games_played_col].values

        # likelihood
        p_win = pm.Binomial("p_win", n=n, p=p, observed=data[outcome_col].values)

        trace = pm.sample()

        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if save_strategy == 'thin':
                thinned_trace = trace.sel(draw=slice(None, None, thinning_factor))
                thinned_trace.to_netcdf(os.path.join(output_path, "bayesian_model_trace.nc"))
            elif save_strategy == 'summary':
                theta_mean = trace.posterior['theta'].mean(dim=('chain', 'draw'))
                theta_sd = trace.posterior['theta'].std(dim=('chain', 'draw'))
                beta_mean = trace.posterior['beta'].mean(dim=('chain', 'draw'))
                beta_sd = trace.posterior['beta'].std(dim=('chain', 'draw'))

                theta_mean.to_netcdf(os.path.join(output_path, "bayesian_theta_mean.nc"))
                theta_sd.to_netcdf(os.path.join(output_path, "bayesian_theta_sd.nc"))
                beta_mean.to_netcdf(os.path.join(output_path, "bayesian_beta_mean.nc"))
                beta_sd.to_netcdf(os.path.join(output_path, "bayesian_beta_sd.nc"))

            # Save metadata and mappings
            meta = {'strategy': save_strategy}
            with open(os.path.join(output_path, 'bayesian_model_meta.json'), 'w') as f:
                json.dump(meta, f)
            
            mappings = {'season_map': season_map, 'team_map': team_map}
            joblib.dump(mappings, os.path.join(output_path, "bayesian_model_mappings.pkl"))

    return trace

def predict_bayesian_team_ability(new_data, model_path, team_col, opponent_col, season_col):
    """
    Makes predictions using a trained Bayesian team ability model.

    Args:
        new_data (pd.DataFrame): New data for prediction. Must contain team_col, opponent_col, and season_col.
        model_path (str): Path to the saved model artifacts (trace or summary) and mappings.
        team_col (str): The name of the column containing team names.
        opponent_col (str): The name of the column containing opponent names.
        season_col (str): The name of the column containing season information.

    Returns:
        pd.DataFrame: The new_data dataframe with added columns for predicted win probability.
    """
    # 1. Load mappings and determine strategy
    mappings = joblib.load(os.path.join(model_path, "bayesian_model_mappings.pkl"))
    season_map = mappings['season_map']
    team_map = mappings['team_map']

    meta_path = os.path.join(model_path, "bayesian_model_meta.json")
    strategy = 'thin' # Default for older models
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        strategy = meta.get('strategy', 'thin')

    # 2. Preprocess new_data
    new_data = new_data.copy()
    for team in pd.concat([new_data[team_col].astype(str), new_data[opponent_col].astype(str)]).unique():
        if team not in team_map:
            raise ValueError(f"Team '{team}' not found in training data. Cannot make prediction.")
    for season in new_data[season_col].unique():
        if season not in season_map:
            raise ValueError(f"Season '{season}' not found in training data. Cannot make prediction.")

    # 3. Make predictions based on strategy
    if strategy == 'thin':
        trace = az.from_netcdf(os.path.join(model_path, "bayesian_model_trace.nc"))
        season_ix = new_data[season_col].map(season_map).values
        team_ix = new_data[team_col].astype(str).map(team_map).values
        opps_ix = new_data[opponent_col].astype(str).map(team_map).values

        theta_posterior = trace.posterior['theta']
        beta_posterior = trace.posterior['beta']

        results = []
        for i in range(len(new_data)):
            s_ix, t_ix, o_ix = season_ix[i], team_ix[i], opps_ix[i]
            theta_s = theta_posterior.isel(season=s_ix, team=t_ix)
            beta_s = beta_posterior.isel(season=s_ix, opps=o_ix)
            logit_s = theta_s - beta_s
            prob_s = 1 / (1 + np.exp(-logit_s.values))
            mean_prob = prob_s.mean()
            hdi = az.hdi(prob_s, hdi_prob=0.94)
            results.append({
                'predicted_win_prob_mean': mean_prob,
                'predicted_win_prob_hdi_3%': hdi[0],
                'predicted_win_prob_hdi_97%': hdi[1]
            })
        predictions_df = pd.DataFrame(results)
        new_data = pd.concat([new_data.reset_index(drop=True), predictions_df], axis=1)

    elif strategy == 'summary':
        theta_mean = xr.open_dataarray(os.path.join(model_path, "bayesian_theta_mean.nc"))
        theta_sd = xr.open_dataarray(os.path.join(model_path, "bayesian_theta_sd.nc"))
        beta_mean = xr.open_dataarray(os.path.join(model_path, "bayesian_beta_mean.nc"))
        beta_sd = xr.open_dataarray(os.path.join(model_path, "bayesian_beta_sd.nc"))

        season_da = xr.DataArray(new_data[season_col].values, dims="match")
        team_da = xr.DataArray(new_data[team_col].astype(str).values, dims="match")
        opps_da = xr.DataArray(new_data[opponent_col].astype(str).values, dims="match")

        theta_mean_vals = theta_mean.sel(season=season_da, team=team_da)
        theta_sd_vals = theta_sd.sel(season=season_da, team=team_da)
        beta_mean_vals = beta_mean.sel(season=season_da, opps=opps_da)
        beta_sd_vals = beta_sd.sel(season=season_da, opps=opps_da)

        n_samples = 400
        theta_samples = np.random.normal(theta_mean_vals.values, theta_sd_vals.values, size=(n_samples, len(new_data)))
        beta_samples = np.random.normal(beta_mean_vals.values, beta_sd_vals.values, size=(n_samples, len(new_data)))

        logit_samples = theta_samples - beta_samples
        prob_samples = 1 / (1 + np.exp(-logit_samples))

        mean_probs = prob_samples.mean(axis=0)
        hdi_probs = az.hdi(prob_samples, hdi_prob=0.94)

        new_data['predicted_win_prob_mean'] = mean_probs
        new_data['predicted_win_prob_hdi_3%'] = hdi_probs[:, 0]
        new_data['predicted_win_prob_hdi_97%'] = hdi_probs[:, 1]

    return new_data