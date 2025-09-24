#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np

import requests

import os
import time
from datetime import datetime
import sys
import yaml
import gc


from data_loader import *
from utils import *
from nn import *
from models import *
from llm import *
from llm_hf import *
import folder_manager

import seaborn as sns
sns.set_style("darkgrid")
plt.rcParams.update({
    'axes.facecolor': '#1e1e1e',
    'figure.facecolor': '#1e1e1e',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.grid': True,
    'grid.color': 'gray'
})

pd.set_option("display.max_column",None)
print(os.getcwd())


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

pd.options.display.max_rows = 100


def auto_reload():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('reload_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# In[2]:


with open("config.yaml",'r') as f:
    config = yaml.safe_load(f)

home_dir = config['HOME_DIRECTORY']
home_dir


# In[3]:


create_submodel("llm_new")


# # API Football:
# https://www.api-football.com

# ![PYTHON LOGO](https://www.api-football.com/public/img/news/archi-beta.jpg)

# In[4]:


leagues_dat = get_leagues(home_dir +"/data/Leagues/leagues.parquet")
leagues_dat[['league_id','league_name','country_name']]


# # Leagues subset:

# In[5]:


# Configs
major_leagues = ["Premier League","La Liga","Serie A","Bundesliga","Eredivisie","Ligue 1"]
major_countries = ["England","Spain","Italy","Germany","Netherlands","France","Brazil"]
teams = ["Liverpool","Wolves"] # teams to pull players data of
seasons = [2022,2021,2023,2024] # seasons to pull players and teams stats of



leagues_subset = leagues_dat[leagues_dat.league_name.isin(major_leagues) & leagues_dat.country_name.isin(major_countries)] # league ID to pull from, current values: {39:premier league}, Add to dictionary as needed


# # Read All fixtures data

# In[6]:


teams_dat = pd.read_parquet(home_dir + "/data/Teams/team_league.parquet")


# In[7]:


fixtures_dir = home_dir + "/data/Fixtures"

complete_data = pd.DataFrame()
for file in os.listdir(fixtures_dir):
    dat = pd.read_parquet(os.path.join(fixtures_dir,file))
    complete_data = pd.concat([complete_data,dat],axis = 0)

complete_data = complete_data.reset_index()
complete_data.drop(columns = ['index'],inplace=True)


# In[8]:


complete_data.columns


# In[9]:


leagues_dat.head()


# In[10]:


# Data checks
complete_data['passes_accuracy'] = complete_data['passes_accuracy'].astype("float64")
complete_data.rename(columns= {'passes_accuracy':'passes_accurate'},inplace =True)
complete_data['fixture_date'] = pd.to_datetime(complete_data.fixture_date)
complete_data['fixture_date_dt'] = complete_data['fixture_date'].dt.date
complete_data = create_datetime_columns(complete_data,'fixture_date')
complete_data['games_rating'] = pd.to_numeric(complete_data['games_rating'])

complete_data['season'] = complete_data['fixture_date'].apply(get_season)


# Targets
complete_data['outcome_num'] = pd.Categorical(complete_data.outcome).codes

complete_data['win'] = np.where(complete_data.outcome.str.lower() == 'win', 1,0)

# Joins:
complete_data = complete_data.merge(teams_dat.drop_duplicates(),how = 'left', left_on= 'team',right_on = 'team_name').drop(columns = ['team_name'])
complete_data = complete_data.merge(leagues_dat[['league_id','league_name']],how = 'left', left_on = 'league', right_on = 'league_id')


# In[11]:


complete_data.head()


# In[12]:


complete_data[complete_data['fixture_date'].isna()]


# In[13]:


# This is the dictionary that contains all information about the features    
dat_dict = find_data_types(complete_data,config['OUTCOME_COLS'] + ['outcome_num','outcome'])
dat_dict = pd.DataFrame(list(dat_dict.items()),columns =['feature','type'])

# differentiate modeling features
non_modeling_features = config['FIXTURE_COLS'] + config['OUTCOME_COLS'] + config['MISC_COLS'] + ['outcome_num','league','win','fixture_date','season','fixture_date_dt','major_position']
dat_dict['modeling_feature'] = np.where(dat_dict['feature'].isin(non_modeling_features),0,1)
dat_dict['encoded'] = 0

print(dat_dict['type'].value_counts())
dat_dict.reset_index(drop= True)

## Encode Features
dat_dict = create_data_index(complete_data,dat_dict,'target',folder_manager.encoding_path)
dat_dict[dat_dict.modeling_feature ==1]


# In[14]:


# primary position map:
player_position = complete_data.groupby(["player_id","games_position"],as_index = False).agg(games_played = ("player_id","size"))
player_position['multiple_records'] = player_position.groupby('player_id')['games_played'].transform("cumsum")
player_position['multiple_records'] = player_position.groupby('player_id')['multiple_records'].transform("max")
player_position['major_position'] = np.where(player_position.games_played/player_position.multiple_records >= .5, player_position.games_position,None)
player_position_map = player_position[['player_id','major_position']].dropna().drop_duplicates()
player_position_map

# Join back to complete_data

complete_data = pd.merge(complete_data,player_position_map,on = 'player_id',how = 'left')


# In[15]:


# Run Player Comparison from LLm 
#player_compare  = compare_players_from_llm(complete_data,["Giovanni Leoni","Ibrahima Konaté"],years = [2025],normalize=True)


# In[ ]:


# Fixture-Player data aggregated to Fixture level:
fixture_dat = calculate_fixture_stats(complete_data,['league_name'])


# In[73]:


# Sanity Check
fixture_dat['fixture_id'].value_counts(ascending = False)


# # Clustering Opponents

# ## Method 1: Decision Tree

# In[ ]:


# team Cluster by tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from category_encoders import OrdinalEncoder


target_col = 'team_goals_conceded'

# Data to train Decision Tree on
# team_class_dat = fixture_dat[['fixture_id','team','opponent','season','win']].drop_duplicates()
team_class_dat = fixture_dat[list(set(['fixture_id','team','opponent','season','league_name',
'games_rating','shots_total','shots_on','passes_key','passes_accurate','duels_total','duels_won','fouls_drawn',
'dribble_success_rate','target_shot_conversion_perc','duels_won_perc','pass_accuracy_perc','fouls_committed','penalty_won'] + [target_col])) ].drop_duplicates()

model_cols = []#['opponent_encoded,'games_rating','shots_total','shots_on','passes_key','passes_accurate','duels_total','duels_won','fouls_drawn','dribble_success_rate','target_shot_conversion_perc','duels_won_perc','pass_accuracy_perc','fouls_committed','penalty_won']



# Encode Non numeric features
oe = OrdinalEncoder()
team_class_dat['team_encoded'] = oe.fit_transform(team_class_dat['team'])
team_class_dat['opponent_encoded'] = oe.transform(team_class_dat.drop(columns = 'team').rename(columns={"opponent":'team'})['team']).astype("int")
team_class_dat.head()
team_class_dat['team_cluster'] = 0

for league, season in team_class_dat[['league_name', 'season']].drop_duplicates().itertuples(index=False):
    if season != np.nan:
        print(f"for season {season}, and league {league}")
        dtc = DecisionTreeClassifier(max_depth=4)
        print("Using cols: ", ['team_encoded'] + model_cols )
        model = dtc.fit(team_class_dat[(team_class_dat.season == season) & (team_class_dat.league_name == league)][['team_encoded'] + model_cols],team_class_dat[(team_class_dat.season == season) & (team_class_dat.league_name == league)][target_col].values)
        y_pred = model.predict(team_class_dat[(team_class_dat.season == season) & (team_class_dat.league_name == league)][['team_encoded']+ model_cols])
        team_class_dat.loc[(team_class_dat.season == season) & (team_class_dat.league_name == league),'team_cluster'] = model.predict_proba(team_class_dat[(team_class_dat.season == season) & (team_class_dat.league_name == league)][['team_encoded']+ model_cols]).max(axis = 1)

team_cluster_map = team_class_dat[['season','league_name','team','team_cluster']].drop_duplicates().reset_index(drop= True)
team_class_dat = team_class_dat.merge(team_cluster_map,left_on = ['season','league_name','opponent'],right_on = ['season','league_name','team'],how = 'left').rename(columns = {'team_cluster_y':'opponent_cluster',
                                                                                                              'team_cluster_x':'team_cluster',
                                                                                                              'team_x':'team'}).drop(columns = ['team_y'])

oe_cluster = OrdinalEncoder()
team_class_dat['opponent_cluster_encoded'] = oe_cluster.fit_transform(team_class_dat['opponent_cluster'].astype("str"))
team_class_dat['team_cluster_encoded'] = oe_cluster.fit_transform(team_class_dat['team_cluster'].astype("str"))

oe_season = OrdinalEncoder()
team_class_dat['season_ix'] = oe_season.fit_transform(team_class_dat['season'])


# In[ ]:


# Check Cluster differences
team_class_dat.groupby('league_name')['opponent_cluster'].apply(lambda x: x.isna().mean())


# # Bayesian Team Ability Estimation
# 
#  - team_ability: alpha ~ Normal(mu,sig^2)
#  - opposition_difficulty: beta ~ Normal(mu,sig^2)
# 
#  - P(w) ~ binomial(N,alpha - beta)

# In[ ]:


# Data for Binomial Model

team_class_dat_binom = team_class_dat.groupby(['season','team','team_cluster_encoded','opponent_cluster_encoded'],as_index = False).agg(wins = ('win','sum'), total_games = ('win','size'))
#Encode season
season_oe = OrdinalEncoder()
team_class_dat_binom['season_encoded'] = season_oe.fit_transform(team_class_dat_binom['season'])
team_class_dat_binom.head()


# In[ ]:


# PYMC model:
# Model per season to estimate win probability given team ability, opponent cluster difficulty  

season_ix_raw = team_class_dat_binom['season']
team_ix_raw = team_class_dat_binom['team']
opps_ix_raw = team_class_dat_binom['opponent_cluster_encoded']

season_map = {x:i for i,x  in enumerate(team_class_dat_binom['season'].unique()) }
team_map = {x:i for i,x  in enumerate(team_class_dat_binom['team'].unique()) }
opps_map = {x:i for i,x  in enumerate(team_class_dat_binom['opponent_cluster_encoded'].unique()) }

season_ix = season_ix_raw.map(season_map).to_numpy()
team_ix = team_ix_raw.map(team_map).to_numpy()
opps_ix = opps_ix_raw.map(opps_map).to_numpy()

coords = {
    "team": team_class_dat_binom.team.unique(),
    "opps" : team_class_dat_binom.opponent_cluster_encoded.unique(),
    "season" : team_class_dat_binom.season.unique()
}



with pm.Model(coords = coords) as model:

    mu_team = pm.Normal("mu_team", 0, 2)
    sigma_team = pm.HalfNormal("sigma_team", 3)

    mu_opps = pm.Normal("mu_opps", 0, 3)
    sigma_opps = pm.HalfNormal("sigma_opps", 5)

    # Raw Ability:
    theta_raw = pm.Normal("theta_raw",0,1, dims = ("season","team"))
    theta_team = mu_team + theta_raw * sigma_team
    theta = pm.Deterministic("theta",theta_team - theta_team.mean(axis = 1,keepdims= True), dims = ("season","team"))

    beta_raw = pm.Normal("beta_raw",0,1,dims = ("season","opps"))
    beta_team = mu_opps + beta_raw * sigma_opps
    beta = pm.Deterministic("beta", beta_team - beta_team.mean(axis = 1,keepdims = True), dims = ("season","opps"))

    logit = theta[season_ix,team_ix] - beta[season_ix,opps_ix]
    p = pm.Deterministic("p",pm.math.sigmoid(logit))
    n = team_class_dat_binom['total_games'].values

    # Likelihood:
    p_win = pm.Binomial("p_win",p = p, n = n,observed = team_class_dat_binom['wins'].values)

    trace = pm.sample(return_inferencedata=True)




# In[ ]:


# Posterior
p_summary =  pm.summary(trace)
p_summary


# In[ ]:


season_filter = ['2022/2023','2024/2025','2023/2024']
teams_filter = complete_data[(complete_data.season == '2024/2025') & (complete_data.league_name == 'Premier League')]['team'].unique()#['Liverpool','Chelsea','Nottingham Forest','Manchester United','Arsenal','Manchester City','Fulham']#complete_data[(complete_data.season == '2024/2025') & (complete_data.league_name == 'Premier League')]['team'].unique()

season_dat = trace.posterior.sel(season = season_filter)
team_dat = season_dat.sel(team = teams_filter)
team_dat


# In[ ]:


import plotly.express as px
import numpy as np

# Select your variable
theta = team_dat['theta'] if 'theta' in team_dat.data_vars else team_dat

chains = theta.chain.values
seasons = theta.season.values
teams = theta.team.values

# We'll build a long "plot-ready" dictionary
plot_data = {
    'value': [],
    'team': [],
    'chain': [],
    'season': []
}

# Loop over coordinates and fill the dictionary
for chain in chains:
    for season in seasons:
        for team in teams:
            y = theta.sel(chain=chain, season=season, team=team).values
            plot_data['value'].extend(y)
            plot_data['team'].extend([team]*len(y))
            plot_data['chain'].extend([chain]*len(y))
            plot_data['season'].extend([season]*len(y))

# Create the interactive KDE plot
fig = px.violin(
    plot_data,
    x='team',
    y='value',
    color='team',
    facet_row='chain',
    facet_col='season',
    box=True,          # optional: show boxplot inside violin
    points='all',      # optional: show all individual points
    hover_data=['team', 'chain', 'season']
)

fig.update_layout(height=300*len(chains), width=2000)
fig.show()


# In[ ]:


team_cluster_map[team_cluster_map.league_name.str.contains("Ligue")]


# In[ ]:


# Inspect the training data:


# In[ ]:


# Check Number of Clusters:
# Low numbers could point to data issues
pd.pivot(team_cluster_map.groupby(['season','league_name'])['team_cluster'].nunique().reset_index(),index = 'league_name', columns = "season")


# # Method 2: Kmeans

# In[89]:


# Fixture-Player data aggregated to Fixture level:
fixture_dat = calculate_fixture_stats(complete_data,['league_name'])


# In[90]:


# Since tree classification is on a single metric,
# I will try a K-means clustering approach to account for all data to cluster teams to get ~10-15 different playing styles

kmeans_cols = list(set(['games_rating','shots_total','shots_on','passes_key','passes_accurate','duels_total','duels_won','fouls_drawn',
'dribble_success_rate','target_shot_conversion_perc','duels_won_perc','pass_accuracy_perc','fouls_committed','penalty_won','penalty_commited']))


fixture_dat = fit_kmeans(fixture_dat,kmeans_cols,k = 15)

fixture_dat = fixture_dat.merge(fixture_dat[['team','fixture_id','cluster']],left_on = ['fixture_id','opponent'],right_on = ['fixture_id','team'],suffixes=("","_opponent_km"),how = 'left').drop(columns = ['team_opponent_km'])
fixture_dat = fixture_dat[fixture_dat.cluster_opponent_km.notna()]
fixture_dat['cluster_opponent_km'] = fixture_dat['cluster_opponent_km'].astype("int")
fixture_dat.head()



# In[ ]:


# Inspect Clusters
clusters = pd.read_csv("/Users/echhitjoshi/Library/Mobile Documents/com~apple~CloudDocs/Work/overperformXG/outputs/models/22_15_36_llm_new/kmeanscluster_centers.csv")
clusters


# In[92]:


print("Calculated Playing styles: ")
fixture_dat.cluster.nunique()


# In[93]:


#Binomial data for Bayesian Model:
model_dat = fixture_dat.groupby(['season','team','cluster_opponent_km'],as_index = False).agg(win = ('win','sum'),games_played = ('win','size'))
model_dat.head()


# In[ ]:


# Bayesian model
season_v = model_dat['season']
team_v = model_dat['team']
opps_v = model_dat['cluster_opponent_km']

season_map = {v:i for i,v in enumerate(season_v.unique())}
team_map = {v:i for i,v in enumerate(team_v.unique())}
opps_map = {v:i for i,v in enumerate(opps_v.unique())}

season_ix = season_v.map(season_map).values
team_ix = team_v.map(team_map).values
opps_ix = opps_v.map(opps_map).values

coords = {
    "season" : model_dat['season'].unique(),
    "team": model_dat['team'].unique(),
    "opps": model_dat['cluster_opponent_km'].unique()
}


with pm.Model(coords = coords) as model:

    # team prior
    mu_team = pm.Normal("mu_team",0, 1)
    sigma_team = pm.HalfNormal("sigma_team",2)

    # opponent prior
    mu_opps = pm.Normal("mu_opps",0,2)
    sigma_opps = pm.HalfNormal("sigma_opps",3)

    # raw parameters
    theta_raw = pm.Normal("theta_raw",0,1,dims = ("season","team"))
    theta_t = mu_team + theta_raw * sigma_team
    theta = pm.Deterministic("theta", theta_t - theta_t.mean(axis = 1,keepdims = True),dims = ("season","team"))

    beta_raw = pm.Normal("beta_raw",0,1,dims = ("season","opps"))
    beta_t = mu_opps + beta_raw * sigma_opps
    beta = pm.Deterministic("beta", beta_t - beta_t.mean(axis = 1,keepdims = True),dims = ("season","opps"))

    # Logit: team ability - opps ability/difficulty
    logit = theta[season_ix,team_ix] - beta[season_ix, opps_ix]
    p = pm.Deterministic("p", pm.math.sigmoid(logit))
    n = model_dat['games_played'].values

    # likelihood
    p_win = pm.Binomial("p_win",n = n,p = p, observed = model_dat['win'].values)

    trace = pm.sample()


# In[49]:


post = pm.summary(trace)


# In[55]:


season_filter = ['2022/2023','2024/2025','2023/2024']
teams_filter = complete_data[(complete_data.season == '2024/2025') & (complete_data.league_name == 'Premier League')]['team'].unique()#['Liverpool','Chelsea','Nottingham Forest','Manchester United','Arsenal','Manchester City','Fulham']#complete_data[(complete_data.season == '2024/2025') & (complete_data.league_name == 'Premier League')]['team'].unique()

season_dat = trace.posterior.sel(season = season_filter)
team_dat = season_dat.sel(team = teams_filter)
team_dat


# In[56]:


import plotly.express as px
import numpy as np

# Select your variable
theta = team_dat['theta'] if 'theta' in team_dat.data_vars else team_dat

chains = theta.chain.values
seasons = theta.season.values
teams = theta.team.values

# We'll build a long "plot-ready" dictionary
plot_data = {
    'value': [],
    'team': [],
    'chain': [],
    'season': []
}

# Loop over coordinates and fill the dictionary
for chain in chains:
    for season in seasons:
        for team in teams:
            y = theta.sel(chain=chain, season=season, team=team).values
            plot_data['value'].extend(y)
            plot_data['team'].extend([team]*len(y))
            plot_data['chain'].extend([chain]*len(y))
            plot_data['season'].extend([season]*len(y))

# Create the interactive KDE plot
fig = px.violin(
    plot_data,
    x='team',
    y='value',
    color='team',
    facet_row='chain',
    facet_col='season',
    box=True,          # optional: show boxplot inside violin
    points='all',      # optional: show all individual points
    hover_data=['team', 'chain', 'season']
)

fig.update_layout(height=300*len(chains), width=2000)
fig.show()


# In[ ]:


model.decision_path(team_class_dat[team_class_dat.year_e == year][['team']])


# In[ ]:


pd.Series(y_pred_proba.max(axis=1)).value_counts()


# In[ ]:


filter = 'games_position.isin(["M","D"])'
target = 'team_goals_scored'

col_subset = [['win','games_rating','shots_total','shots_on','goals_total','goals_saves','duels_won']]

cor_dat = fixture_dat.query(filter).corr(numeric_only=True)[[target]]
cor_dat.drop(target,inplace = True)

sorted_cols = cor_dat.sort_values(target,ascending = False).index.to_list()

fig, ax = plt.subplots(1,1,figsize = (15,10))
sns.heatmap(cor_dat.loc[sorted_cols],cmap = 'coolwarm',ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation =75)
fig.show()


# In[ ]:


find_player(complete_data,player_name="Leoni")


# In[ ]:


all_defenders_2025 = complete_data[(complete_data.major_position == 'D') & (complete_data.year_e == 2025)]['player_name'].unique()
all_defenders_2025


# In[ ]:


defenders_compare = compare_players(complete_data,all_defenders_2025,years = [2025],transpose = False)


# In[ ]:


per_90_cols = [col for col in defenders_compare.columns if "per_90" in col]
attack_per_90_cols  = ['total_shots_per_90','shots_on_target_per_90','goals_scored_per_90','assists_per_90',
                       'fouls_drawn_per_90','attempted_dribbles_per_90','successful_dribbles_per_90',
                       'dribble_success_rate_per_90','duels_contested_per_90','duels_won_per_90','duels_won_percentage_per_90']
defense_per_90_cols = ['yellow_cards_per_90','red_cards_per_90','fouls_drawn_per_90','fouls_committed_per_90',
                       'dribbled_past_per_90', 'total_tackles_per_90','blocks_per_90','interceptions_per_90',
                       'duels_contested_per_90','duels_won_per_90','duels_won_percentage_per_90','penalties_committed_per_90']
pass_per_90_cols = [ 'total_passes_per_90','key_passes_per_90', 'average_passes_accurate_per_90','average_pass_accuracy_per_90']


# In[ ]:


defenders_compare.head()


# In[ ]:


# Calculate Clusters:
filter = 'total_minutes_played  > 1000'
defense_cluster = 'defense_cluster'
pass_cluster = 'pass_cluster'
defenders_compare_w_cluster = fit_kmeans(defenders_compare.query(filter),defense_per_90_cols,None,cluster_name)
defenders_compare_w_cluster = fit_kmeans(defenders_compare_w_cluster.query(filter),pass_per_90_cols,None,pass_cluster)


# In[ ]:


find_player(complete_data,player_name="Virgil van")


# In[ ]:


defenders_compare_w_cluster[defenders_compare_w_cluster.player_name.str.contains("William Saliba")]


# In[ ]:


defense_cluster


# In[ ]:


clusters = defenders_compare_w_cluster[defenders_compare_w_cluster.player_name.str.contains('William Saliba')][[defense_cluster,pass_cluster]].values
clusters
#defenders_compare_w_cluster[defenders_compare_w_cluster[cluster_name].isin(defenders_compare_w_cluster[condition][cluster_name])].sort_values("average_rating",ascending = False)


# In[ ]:


folder_manager.llm_code_path


# In[ ]:


question = "How are you doing?"
question_no_spec = re.sub(r"[?.,;:]","",question)
split_words = [word for word in question_no_spec.split(" ")]
split_words


# In[ ]:


complete_data.columns


# In[ ]:


defenders_compare_w_cluster[(defenders_compare_w_cluster.player_name.isin(["Mike Eerdhuijzen","Giovanni Leoni","Nikola Milenković","Marc Guéhi","Ladislav Krejčí"]))][['player_name'] + [col for col in defenders_compare.columns if "per_90" in col]].T


# In[ ]:


schema = {
        "columns": list(complete_data.columns),
        "nrows": [complete_data.shape[0]],
        "dtypes": {col : str(complete_data[col].dtype) for col in complete_data.columns}
    }


# In[ ]:


complete_data.columns


# In[ ]:


plot_from_llm(complete_data[complete_data.player_name == 'Olivier Boscagli'],"Plot Average games_rating with error cloud by month_e faceted by team")


# In[ ]:


plot_continuous_trend(complete_data[complete_data.player_name == 'Emmanuel Agbadou'],"month_e","games_rating")


# In[ ]:


filter_query = 'major_position.isin(["M"])'


# Stat to look at:
stat = 'target_shot_conversion_perc'
agg_fun = "mean"
rank_cutoff = 20

# configs 
min_appearance = 40

dribble_dat_g = complete_data.query(filter_query).reset_index().fillna(0).groupby("player_name").agg(n_apps = ("player_name","size"),stat = (stat,agg_fun)).reset_index()
dribble_dat_g = dribble_dat_g[dribble_dat_g.n_apps >= min_appearance]
dribble_dat_g['rank'] = dribble_dat_g["stat"].fillna(0).rank(ascending= False,method = 'dense')
dribble_dat_g.sort_values("rank",inplace = True)

fig, ax = plt.subplots(figsize=(13, 8))

# Plot correctly, no comma here
sns.boxplot(
    data=complete_data.query(filter_query)[complete_data.query(filter_query).player_name.isin(dribble_dat_g[dribble_dat_g['rank'] < rank_cutoff]['player_name'])],
    x="player_name",
    y=stat,
    order=dribble_dat_g[dribble_dat_g['rank'] < rank_cutoff]['player_name'],
    ax=ax,
    
)

# Now this works correctly on `ax`
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title(f"Stat: {stat}")
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.subplots(nrows=1, ncols = 1, figsize = (20,10))
fig = sns.heatmap(complete_data.query(filter_query)[config['PASSING_COLS']  + ['team_goals_scored','team_non_penalty_goals_scored','team_goals_conceded']].corr(),cmap = 'coolwarm')
fig.set_xticklabels(fig.get_xticklabels(),rotation = 60)


# In[ ]:


sns.pairplot(complete_data.query(filter_query)[config['PASSING_COLS']  + ['team_goals_scored','team_non_penalty_goals_scored','team_goals_conceded']])


# In[ ]:


complete_data.columns


# In[ ]:


config['PASSING_COLS'] + config['DEFENSE_COLS']


# In[ ]:


# trial multiclass model:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(complete_data.query(filter_query)[list(set(config['PASSING_COLS'] + config['DEFENSE_COLS'] ))],
                                                    complete_data.query(filter_query)['win'],
                                                    stratify=complete_data.query(filter_query)['win'],
                                                    random_state=33)


# In[ ]:


create_submodel("catboost")


# In[ ]:


output_path


# In[ ]:


model = run_model_with_fs_tune(X_train, X_test, y_train, y_test,dat_dict,'catboost',output_path=folder_manager.output_path)


# In[ ]:


dat = NNDataFromPd(X_train.fillna(0), y_train.outcome_num, dat_dict)
train_loader = DataLoader(dat, batch_size = 128,shuffle= True)


# In[ ]:


train_loader.dataset.X_numeric_tensor.shape


# In[ ]:


# model params
n_features = X_train.shape[1]
n_classes = y_train.iloc[:,0].nunique()
model = MultiClassModel(n_features,n_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)


# In[ ]:


epochs = 500

for epoch in range(epochs):
    
    epoch_loss = 0

    for X_numeric_batch, X_categoric_batch, y_batch in train_loader:
        
        pred = model.forward(X_numeric_batch)
        
        loss = criterion(pred,y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch: {epoch}, Loss: {epoch_loss}")


# In[ ]:


model.eval()


# In[ ]:


X_test['passes_accuracy'] = X_test['passes_accuracy'].astype("float64")


# In[ ]:


# test sets

test_dat = NNDataFromPd(X_test,y_test,dat_dict)
test_loader = DataLoader(test_dat,batch_size= X_test.shape[0],shuffle=True)


# In[ ]:


model.eval()
with torch.no_grad():
    for X_numeric_batch, X_categoric_batch, y_batch in test_loader:
        output = model(X_numeric_batch)
        pred_class = torch.argmax(output, dim = 1)


# In[ ]:


# Logistic Model:
X_train, X_test, y_train, y_test = train_test_split(complete_data[complete_data.games_position == 'F'][list(set(config['DEFENSE_COLS'] + config['PASSING_COLS'] + config['ATTACK_COLS'])) + ['win']].drop(columns = 'win'),
                                                    complete_data[complete_data.games_position == 'F']['win'],
                                                    stratify=complete_data['win'],
                                                    random_state=33)


# In[ ]:





# In[ ]:


train_dat = NNDataFromPd(X_train,y_train,dat_dict)
train_loader = DataLoader(train_dat,batch_size= 128,shuffle = True)


# In[ ]:


n_features = X_train.shape[1]
model = LogisticNNModelComplex(n_features)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.005)


# In[ ]:


epochs = 500
for epoch in range(epochs):
    epoch_loss = 0
    
    for X_numeric, X_categoric, y in train_loader:

        pred = model(X_numeric)

        loss = criterion(pred,y.unsqueeze(1))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch},  Loss: {loss}")


# In[ ]:


pred_proba.squeeze(1)


# In[ ]:


from validations import *


test_dat = NNDataFromPd(X_test,y_test,dat_dict)
test_loader = DataLoader(test_dat,batch_size= X_test.shape[0],shuffle=True)

model.eval()
with torch.no_grad():
    for X_numeric_batch, X_categoric_batch, y_batch in test_loader:
        output = model(X_numeric_batch)
        pred_proba = torch.softmax(output,dim =1)
        pred_class = torch.argmax(output, dim = 1)


discrete_evaluations(y_test,pred_class,pred_proba.squeeze(1),classification_type="Binary",model_path= folder_manager.output_path)


# In[ ]:


test_fixtures = get_team_fixtures("Liverpool",2)


# In[ ]:


test_fixtures


# In[ ]:


player_stat_url = "https://v3.football.api-sports.io/fixtures/players?fixture={}".format(1035045)
fixture_dat = requests.get(player_stat_url,headers=headers_api_sport)


# In[ ]:


pd.json_normalize(pd.json_normalize(fixture_dat.json()['response']))['players'][0]


# In[ ]:


fixture_dat_expanded = pd.concat([pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])[['player.id','player.name']],pd.json_normalize(pd.json_normalize(pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])['statistics']).rename(columns = {0:"player_stats"})['player_stats'])],axis = 1)


# In[ ]:


fixtures_stat = complete_data.groupby(['fixture_id','team'],as_index=False).agg(n_opponent = ('opponent','count'),total_passes = ('passes_total','sum')).sort_values('fixture_id',ascending= False)


# In[ ]:


fixtures_stat


# In[ ]:


complete_data[complete_data.fixture_id == 1376437][['team','opponent']]


# In[ ]:


teams_dat[teams_dat.team_name.str.contains("Tels")]


# In[ ]:


angers = pd.read_parquet(home_dir + "/data/Fixtures/angers_2024.parquet")


# In[ ]:


angers['fixture_date'] = pd.to_datetime(angers['fixture_date'])


# In[ ]:


angers['fixture_date']


# In[ ]:




