#!/usr/bin/env python
# coding: utf-8

# In[27]:


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



fixture_dat = calculate_fixture_stats(complete_data,['league_name'])

kmeans_cols = list(set(['games_rating','shots_total','shots_on','passes_total','passes_key','passes_accurate','duels_total','duels_won','fouls_drawn','cards_yellow','tackles_interceptions','tackles_blocks',
'dribble_success_rate','dribbles_past','target_shot_conversion_perc','duels_won_perc','pass_accuracy_perc','fouls_committed','fouls_drawn','penalty_won','penalty_commited']))


fixture_dat = fit_kmeans(fixture_dat,kmeans_cols,k = 15)

fixture_dat = fixture_dat.merge(fixture_dat[['team','fixture_id','cluster_rank']],left_on = ['fixture_id','opponent'],right_on = ['fixture_id','team'],suffixes=("","_opponent_km"),how = 'left').drop(columns = ['team_opponent_km'])
fixture_dat = fixture_dat[fixture_dat.cluster_rank_opponent_km.notna()]
fixture_dat['cluster_rank'] = fixture_dat['cluster_rank'].astype("int")
fixture_dat['cluster_rank_opponent_km'] = fixture_dat['cluster_rank_opponent_km'].astype("int")
fixture_dat.head()



# Clusters grouped by win rate
cluster_map = fixture_dat.groupby(['cluster','cluster_rank'],as_index = False).agg(games = ('cluster','size') )
clusters = pd.read_csv(config['HOME_DIRECTORY'] + "/outputs/models/" +folder_manager.submodel_name+"/kmeanscluster_centers.csv").reset_index().rename(columns = {'index':'cluster'})
clusters = clusters.merge(cluster_map,how = 'left').drop(columns = ['cluster'])
print(f"Number of clusters: {clusters.shape[0]}")
clusters = pd.concat([clusters.iloc[:,-2:], clusters.iloc[:,:-2]],axis = 1).sort_values('cluster_rank')
clusters['cluster_rank'] = clusters['cluster_rank'].astype("int")
clusters.sort_values('cluster_rank',ascending = True,inplace = True)
clusters
