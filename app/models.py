import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
import pytensor

from datetime import datetime, timedelta
import os
import glob
import yaml
import joblib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed

from visualizations import *
from validations import discrete_evaluations, check_feature_importance, tune_prob_threshold
from reports import *

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

with open("config.yaml","r") as file:
    config = yaml.safe_load(file)

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


def fit_kmeans(dat,target:list,k = None,cluster_colname = 'cluster'):
    '''
    dat: data with target col
    target: column name
    '''


    if not k:
        k = range(1,25)

        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(check_elbow,i,dat[target].dropna(axis = 0)): i for i in k}

            results = {}
            for future in futures:
                k_val = futures[future]
                kmeans_val = future.result()
                results[k_val] = kmeans_val.inertia_        

        ks = sorted(results.keys())
        inertias = [results[k] for k in ks]
        
        first_deriv = np.diff(inertias)/np.diff(ks)
        second_deriv = np.diff(first_deriv)

        elbow_idx = np.argmin(np.abs(second_deriv)) + 1
        elbow_k = ks[elbow_idx]

        print(f"lowest inertia value change: {elbow_k}")


        plt.plot(ks, inertias, marker='o')
        plt.axhline(y = inertias[elbow_k - 1],
                    color = 'r',
                    linestyle = '--',
                    label = f'Elbow at k = {elbow_k}')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Curve')
        plt.show()

        k = elbow_k


    final_kmeans = KMeans(k,random_state=33)
    print(f"dropping {dat.shape[0] - dat.dropna().shape[0]} players due to nulls! ")
    dat.dropna(axis = 0,inplace = True)
    dat[f"{cluster_colname}"] = final_kmeans.fit_predict(dat[target])

    return dat

    








