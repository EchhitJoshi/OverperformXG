import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.utils import resample
import category_encoders as ce
import joblib
import os
import yaml
from itables import show

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly"
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sc
import requests
from pandas.api.types import is_datetime64_any_dtype, is_period_dtype

from data_loader import get_leagues
import folder_manager


#Themes and options
px.defaults.template = 'plotly_dark'
pd.options.display.max_columns = 200
sns.set_style("ticks")
plt.style.use("dark_background")


config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

home_dir = config['HOME_DIRECTORY']
headers_api_sport = config["HEADERS_API_SPORT"]


def get_major_position(dat):
    player_position = dat.groupby(["player_id","games_position"],as_index = False).agg(games_played = ("player_id","size"))
    player_position['multiple_records'] = player_position.groupby('player_id')['games_played'].transform("cumsum")
    player_position['multiple_records'] = player_position.groupby('player_id')['multiple_records'].transform("max")
    player_position['major_position'] = np.where(player_position.games_played/player_position.multiple_records >= .5, player_position.games_position,None)
    player_position_map = player_position[['player_id','major_position']].dropna().drop_duplicates()
    return player_position_map

def check_feature_information(dat,col):
    '''
    dat: data
    col: column to get more information about
    '''
    print("data type: ",dat[[col]].info())
    print("\n")
    print("descriptive stats: \n",dat[col].describe(include = 'all'))
    print("\n")
    print("null %: ", dat[col].isna().mean() * 100)
    print("\n")

    # distribution:
    if dat[[col]].dtypes.values == "object":
        print(dat[col].value_counts(normalize = True))
        # plot hist
        sns.histplot(dat[col])
        plt.show()
    elif dat[[col]].dtypes.values in('float64','int64') :
        # if many unique, it is possibly continuous
        if dat[col].nunique()/len(dat[col]) > .2:
            #plot density
            sns.kdeplot(dat[col])
            plt.show()
        else:
            # plot hist
            sns.histplot(dat[col])
            plt.show()
    
def find_columns(dat,col_name):
    '''
    dat: pandas df
    col_name: column name to search for
    Returns columns with names including col_name
    '''
    return [col for col in dat.columns if col_name in col]

def infer_target_type(dat,target):

    problem_type = 'regression' if pd.api.types.is_float_dtype(dat[target]) or \
                                      (pd.Series(dat[target]).nunique() > 20 and pd.api.types.is_numeric_dtype(dat[target])) \
                                   else 'classification'
    return problem_type

def check_feature_relation(dat,col:str,target:str):
    '''
    dat: data
    col: column_name to check relation with a target
    target: column_name to check relation with a specified feature
    '''

    unique_targets = dat[target].unique()
    ## samples:
    samples = {}
    for val in unique_targets:
        samples[val] =  dat[dat[target] == val][col].dropna() 

    # 
    target_prop = pd.DataFrame(dat[target].value_counts(normalize=True))
    
    # distribution:
    if dat[[col]].dtypes.values == "object":
        print(dat[col].value_counts(normalize = True))
        # plot hist
        fig = px.histogram(dat,x = col,facet_row=target,nbins = dat[col].nunique() + 1)
        fig.show()
    elif dat[[col]].dtypes.values in('float64','int64') :
        # if many unique, it is possibly continuous
        if dat[col].nunique()/len(dat[col]) > .15:
            
            ### Quick non parametric Kolmogorov-Smirnov test:
            stat,p = sc.ks_2samp(samples[list(samples.keys())[0]],samples[list(samples.keys())[1]])
            print(f"p_value from Non parametric KS-test: {p:.4f}",)
            
            #plot density
            fig = px.histogram(dat,x = col,facet_row= target,histnorm= 'density',nbins = 200)
            fig.show()
        else:
            # plot hist
            fig_hist = px.histogram(dat,x = col,color=target,nbins = dat[col].nunique() + 1)
            fig_scatter = px.scatter(dat,x = col, y = target)
            fig = make_subplots(rows = 1, cols = 2)
            # Add histogram traces
            for trace in fig_hist.data:
                fig.add_trace(trace, row=1, col=1)

            # Add scatterplot traces
            for trace in fig_scatter.data:
                fig.add_trace(trace, row=1, col=2)
            
            fig.update_layout(title = "histogram and scatter relations",showlegend = True)
            fig.show()
            

def find_data_types(dat,target_cols):
    '''
    dat: data to check types
    returns data type(categorical or numeric)
    '''
    dat_type = dict()
    for col in dat.columns:
        # Objects are discrete
        if col in target_cols:
            dat_type[col] = 'target'
            continue
        if dat[col].dtype == 'O' or dat[col].dtype.name == 'category':
            dat_type[col] = 'categorical'
        elif dat[col].dtype in ['int64','float64']:
            dat_type[col] = 'numeric'
        elif is_datetime64_any_dtype(dat[col]) or is_period_dtype(dat[col]):
            dat_type[col] = 'datetime'
        else:
            dat_type[col] = 'other'

    return dat_type
def add_smalls(dat,data_dict,smalls_threshold):
    if smalls_threshold > 1:
        smalls_threshold = smalls_threshold/len(dat)
    feature_proportion = dict()
    for col in dat[data_dict[data_dict['type'] == 'categorical']['feature']].columns:
        feature_proportion[col] = dat[col].value_counts(normalize = True).reset_index().rename(columns = {"index":"feature"})
        feature_proportion[col]['smalls_flag'] = np.where(feature_proportion[col]['proportion'] < smalls_threshold,1,0)

    return feature_proportion

def remove_correlated_features_with_nullity(df, threshold=0.8):
    
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    null_counts = df_numeric.isnull().sum()
    
    to_drop = set()
    
    for col in upper.columns:
        correlated_features = upper.index[upper[col] > threshold].tolist()
        
        for corr_feat in correlated_features:
            if corr_feat in to_drop or col in to_drop:
                continue
            
            if null_counts[col] < null_counts[corr_feat]:
                to_drop.add(col)
            else:
                to_drop.add(corr_feat)
    
    print(f"Features to drop (corr > {threshold}, considering nullity): {sorted(to_drop)}")
    
    reduced_df = df.drop(columns=to_drop)
    return reduced_df, list(to_drop)

def create_data_index(data,data_dict,encoder = 'ordinal',encoding_path = '') -> pd.DataFrame:
    """
    data: pandas df
    encoder: Select from 'ordinal', 'label', 'target'
    """
    columns = data_dict[(data_dict.type == 'categorical') & (data_dict.modeling_feature == 1)]['feature']
    target = data_dict[(data_dict.type == 'target')][['feature']]
    if target.shape[0]>1:
        for feat in target.feature.values:
            if data[feat].dtype == 'int64':
                target = feat

    ordinal_encodings = dict()
    
    for col in columns:
        if encoder.lower() == 'ordinal':
            encoder_alg = ce.ordinal.OrdinalEncoder(handle_unknown='value')
            data[col + "_ix"] = encoder_alg.fit_transform(data[col])
            
        elif encoder.lower() == 'target':
            encoder_alg = ce.target_encoder.TargetEncoder(handle_unknown = 'error')
            data[col + "_ix"] = encoder_alg.fit_transform(data[col],data[target])

        data_dict.loc[data_dict.feature == col, 'encoded'] = 1
        
        ordinal_encodings[col] = encoder_alg
        if encoding_path:
            joblib.dump(encoder_alg,folder_manager.encoding_path+col+"_"+encoder+".pkl")
        
    
    return data_dict

def find_leagues(leagues_dat,league_name= None,country= None):
    if league_name and country:
        return leagues_dat[(leagues_dat.league_name.str.contains(league_name)) & (leagues_dat.country_name.str.contains(country))] 
    elif league_name:
        return leagues_dat[(leagues_dat.league_name.str.contains(league_name)) ] 
    elif country:
        return leagues_dat[(leagues_dat.country_name.str.contains(country)) ] 
    else:
        print("No filter applied")
        return leagues_dat

def refresh_team_league_map(leagues_subset):
    """
    leagues_subset: subset from leagues data
    """
    team_league_map = pd.DataFrame()
    seasons = [2025,2024,2023,2022,2021,2020]
    for season in seasons:
        for league in leagues_subset.league_id:
            print(f"Running for League: {league}, Season: {season}")
            try:
                teams_data_req = requests.get("https://v3.football.api-sports.io/teams?league={}&season={}".format(league,season),headers=headers_api_sport)
                teams = pd.json_normalize(teams_data_req.json()['response'])
                teams = lower_columns(teams)
                if 'team_id' in teams.columns and 'team_name' in teams.columns:
                    team_season = teams[['team_id', 'team_name']].drop_duplicates().copy()
                    team_season['league'] = league
                    team_league_map = pd.concat([team_league_map,team_season],axis = 0)
                else:
                    print(f"Skipping league {league} season {season} due to missing 'team_id' or 'team_name'")
            except Exception as e:
                print(f"Error for League: {league}, Season: {season}, Please check data. Error: {e}")
    
    if not team_league_map.empty:
        team_league_map.drop_duplicates(inplace=True)
        team_league_map.to_parquet(home_dir + "/data/Teams/team_league.parquet")


def find_team(team_name):
    team_league_map = pd.read_parquet(home_dir + "/data/Teams/team_league.parquet")
    return team_league_map[team_league_map.team_name.str.lower().str.contains(team_name.lower())]

def find_player(data,player_id = None,player_name = None):
    if player_id:
        return data[data.player_id == player_id][['player_id','player_name','team']].drop_duplicates()
    else:
        return data[data.player_name.str.contains(player_name)][['player_id','player_name','team']].drop_duplicates()
    
def create_submodel(model_name:str):
    submodel_path = home_dir + "/outputs/models/"
    author = "EJ"
    
    folder_manager.submodel_name = datetime.now().strftime("%d_%H_%M") + "_"+model_name
    folder_manager.output_path = submodel_path+folder_manager.submodel_name
    folder_manager.encoding_path = folder_manager.output_path+"/encodings/"
    folder_manager.feature_report_path = folder_manager.output_path+"/feature_report/"
    
    os.mkdir(folder_manager.output_path)
    os.mkdir(folder_manager.encoding_path)
    os.mkdir(folder_manager.feature_report_path)

def get_latest_model_folder(path):
    folders = [os.path.join(path, d) for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path, d))]
    latest = max(folders, key=os.path.getctime)
    return latest

def get_season(date):
    year = date.year
    if date.month >= 8:
        return f"{year}/{year+1}"
    else:
        return f"{year-1}/{year}"

