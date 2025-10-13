import streamlit as st
import yaml
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import joblib
import scipy.stats as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from data_loader import *
from visualizations import *
from utils import * 
from models import *

#Load configuration
with open("streamlit/app_config.yaml", 'r') as file:
    config = yaml.safe_load(file)

with open(os.path.join(os.path.dirname(__file__), '..', 'app', 'config.yaml'), 'r') as f:
    app_config = yaml.safe_load(f)
home_dir = app_config['HOME_DIRECTORY']

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    
    #Data Reads
    leagues_dat = pd.read_sql("select * from overperformxg.league", con = app_config['MYSQL_STRING'])
    teams_dat = pd.read_sql("select * from overperformxg.team_league_map", con = app_config['MYSQL_STRING'])
    complete_data = pd.read_sql("select * from overperformxg.complete_data", con = app_config['MYSQL_STRING'])
    player_position_map = get_major_position(complete_data)
    
    # Create player stats df
    player_stats_df = compare_players(complete_data, complete_data['player_name'].unique(), complete_data['season'].unique(), transpose=False)
    player_stats_df = player_stats_df.merge(player_position_map, on='player_id', how='left')

    # Calculate percentile ranks
    per_90_stats = [col for col in player_stats_df.columns if '_per_90' in col]
    for stat in per_90_stats:
        player_stats_df[f'{stat}_percentile'] = player_stats_df.groupby(['major_position', 'season'])[stat].rank(pct=True) * 100

    # Get player clusters
    player_stats_df = train_kmeans_and_get_clusters(player_stats_df, n_clusters=50)

    # Deployed model path
    model_path = os.path.join(home_dir, "deployed_models", config['deployed_model'])
    fixture_dat_with_preds = pd.DataFrame()
    clusters = pd.DataFrame()
    
    if os.path.exists(model_path):
        # Kmeans clustering
        fixture_dat = calculate_fixture_stats(complete_data, ['league_name'])
        
        kmeans_model_path = os.path.join(model_path, "kmeans", "kmeans_model.pkl")
        kmeans_features_path = os.path.join(model_path, "kmeans", "kmeans_features.pkl")

        if not os.path.exists(kmeans_features_path):
                # Fallback for older models that don't have saved features
                print("Warning: kmeans_features.pkl not found. Using a hardcoded list of features. This may lead to errors if the model was trained with a different feature set.")
                kmeans_cols = sorted(['games_rating', 'shots_total', 'shots_on', 'passes_total', 'passes_key', 'passes_accurate', 'duels_total', 'duels_won', 'fouls_drawn', 'cards_yellow', 'tackles_interceptions', 'tackles_blocks', 'dribble_success_rate', 'dribbles_past', 'target_shot_conversion_perc', 'duels_won_perc', 'pass_accuracy_perc', 'fouls_committed', 'penalty_won', 'penalty_commited'])
        else:
                kmeans_cols = joblib.load(kmeans_features_path)

        kmeans_model = joblib.load(kmeans_model_path)

        fixture_dat.dropna(subset=kmeans_cols, inplace=True)
        
        # Ensure columns are in the correct order
        fixture_dat_for_prediction = fixture_dat[kmeans_cols]
        
        labels = kmeans_model.predict(fixture_dat_for_prediction)
        fixture_dat['cluster'] = labels

        cluster_win = fixture_dat.groupby('cluster')['win'].mean().reset_index(name='win_perc')
        cluster_win['cluster_rank'] = cluster_win['win_perc'].rank(ascending=False).astype(int)
        
        fixture_dat = fixture_dat.merge(cluster_win[['cluster', 'cluster_rank']], on='cluster')

        # self merge for opponent cluster
        fixture_dat = fixture_dat.merge(fixture_dat[['team','fixture_id','cluster_rank']],left_on = ['fixture_id','opponent'],right_on = ['fixture_id','team'],suffixes=("","_opponent_km"),how = 'left').drop(columns = ['team_opponent_km'])
        fixture_dat.dropna(subset=['cluster_rank_opponent_km'], inplace=True)
        fixture_dat['cluster_rank'] = fixture_dat['cluster_rank'].astype("int")
        fixture_dat['cluster_rank_opponent_km'] = fixture_dat['cluster_rank_opponent_km'].astype("int")

        # Bayesian predictions
        fixture_dat_with_preds = predict_bayesian_team_ability(fixture_dat, model_path, 'team', 'opponent', 'season')

        # Cluster centers
        clusters_df = pd.read_csv(os.path.join(model_path, "kmeanscluster_centers.csv"))
        clusters_df['cluster'] = clusters_df.index
        
        cluster_map = fixture_dat_with_preds.groupby(['cluster','cluster_rank'],as_index = False).agg(games = ('cluster','size'))
        
        clusters = clusters_df.merge(cluster_map, on='cluster')

    return complete_data, player_stats_df, fixture_dat_with_preds, clusters, model_path

df, player_stats_df, fixture_dat_with_preds, clusters, model_path = load_data()

# Create a list of menu item titles
menu_titles = [item['title'] for item in config['menu_items']]

# Create tabs
tabs = st.tabs(menu_titles)

# Populate tabs with content
for i, tab in enumerate(tabs):
    with tab:
        selected_item_config = config['menu_items'][i]

        for subitem in selected_item_config.get('subitems', []):
            if subitem['type'] == 'header':
                st.header(subitem['content'])

            elif subitem['type'] == 'player_selector':
                seasons = st.multiselect('Select Seasons', player_stats_df['season'].unique(), key='player_comparison_seasons')
                players = st.multiselect(subitem['label'], player_stats_df['player_name'].unique(), key=subitem['key'])
                if players and seasons:
                    player_data = player_stats_df[(player_stats_df['player_name'].isin(players)) & (player_stats_df['season'].isin(seasons))]
                    
                    st.subheader("Player Summary")
                    summary_cols = st.columns(len(players))
                    for i, player in enumerate(players):
                        with summary_cols[i]:
                            player_summary_data = player_data[player_data['player_name'] == player]
                            total_games = player_summary_data['total_games_played'].sum()
                            total_minutes = player_summary_data['total_minutes_played'].sum()
                            st.metric(label=f"Games Played ({player})", value=int(total_games))
                            st.metric(label=f"Minutes Played ({player})", value=int(total_minutes))

                    # Define stat categories
                    attacking_stats_per_90 = [
                        'total_shots_per_90_percentile', 'shots_on_target_per_90_percentile',
                        'goals_scored_per_90_percentile', 'goals_converted_per_shot_target_per_90_percentile',
                        'assists_per_90_percentile', 'attempted_dribbles_per_90_percentile',
                        'successful_dribbles_per_90_percentile', 'dribble_success_rate_per_90_percentile',
                        'penalties_won_per_90_percentile', 'penalties_scored_per_90_percentile',
                        'fouls_drawn_per_90_percentile'
                    ]
                    passing_stats_per_90 = [
                        'total_passes_per_90_percentile', 'key_passes_per_90_percentile',
                        'average_passes_accurate_per_90_percentile', 'average_pass_accuracy_per_90_percentile'
                    ]
                    defensive_stats_per_90 = [
                        'total_tackles_per_90_percentile', 'blocks_per_90_percentile',
                        'interceptions_per_90_percentile', 'duels_contested_per_90_percentile',
                        'duels_won_per_90_percentile', 'duels_won_percentage_per_90_percentile',
                        'dribbled_past_per_90_percentile', 'fouls_committed_per_90_percentile',
                        'yellow_cards_per_90_percentile', 'red_cards_per_90_percentile',
                        'penalties_committed_per_90_percentile'
                    ]
                    goalkeeping_stats_per_90 = ['penalties_saved_per_90_percentile']

                    # Create labels
                    attacking_stats_per_90_labels = [s.replace('_percentile', '') for s in attacking_stats_per_90]
                    passing_stats_per_90_labels = [s.replace('_percentile', '') for s in passing_stats_per_90]
                    defensive_stats_per_90_labels = [s.replace('_percentile', '') for s in defensive_stats_per_90]
                    goalkeeping_stats_per_90_labels = [s.replace('_percentile', '') for s in goalkeeping_stats_per_90]

                    # Get major position of the first player
                    major_position = player_data['major_position'].iloc[0]

                    if major_position == 'G':
                        st.subheader("Goalkeeping Stats (per 90)")
                        fig = plot_player_comparison(player_data, players, goalkeeping_stats_per_90, goalkeeping_stats_per_90_labels)
                        st.plotly_chart(fig)
                    else:
                        st.subheader("Attacking Stats (per 90)")
                        fig = plot_player_comparison(player_data, players, attacking_stats_per_90, attacking_stats_per_90_labels)
                        st.plotly_chart(fig)

                        st.subheader("Passing Stats (per 90)")
                        fig = plot_player_comparison(player_data, players, passing_stats_per_90, passing_stats_per_90_labels)
                        st.plotly_chart(fig)

                        st.subheader("Defensive Stats (per 90)")
                        fig = plot_player_comparison(player_data, players, defensive_stats_per_90, defensive_stats_per_90_labels)
                        st.plotly_chart(fig)

                    st.subheader("Stats Comparison")
                    all_stats = list(player_data.columns)
                    # Exclude non-stat columns for selection
                    excluded_cols = ['player_id', 'player_name', 'season', 'major_position', 'cluster']
                    selectable_stats = [s for s in all_stats if s not in excluded_cols]
                    per_90_percentile_stats = [s for s in selectable_stats if '_per_90_percentile' in s]
                    selected_stats = st.multiselect("Select stats to compare", selectable_stats, default=per_90_percentile_stats)
                    if selected_stats:
                        selected_stats_labels = [s.replace('_percentile', '') for s in selected_stats]
                        fig = plot_player_stats_bar_chart(player_data, players, selected_stats, selected_stats_labels)
                        st.plotly_chart(fig)

                    st.subheader("Similar Players")
                    for player in players:
                        st.write(f"---")
                        st.write(f"##### Similar players for {player}:")
                        for season in sorted(seasons, reverse=True):
                            st.write(f"**In season {season}:**")
                            
                            player_season_data = player_data[
                                (player_data['player_name'] == player) & 
                                (player_data['season'] == season)
                            ]

                            if not player_season_data.empty:
                                cluster_id = player_season_data.iloc[0]['cluster']
                                
                                if cluster_id == -1:
                                    st.write("Not enough players in this season to form clusters.")
                                    continue

                                similar_players_df = player_stats_df[
                                    (player_stats_df['season'] == season) &
                                    (player_stats_df['cluster'] == cluster_id) &
                                    (player_stats_df['player_name'] != player)
                                ]
                                
                                if not similar_players_df.empty:
                                    top_5_similar = similar_players_df.sort_values('average_rating', ascending=False).head(5)
                                    
                                    for _, row in top_5_similar.iterrows():
                                        st.write(f"- {row['player_name']} (Avg. Rating: {row['average_rating']:.2f})")
                                else:
                                    st.write("No other similar players found for this season.")
                            else:
                                st.write("Player has no data for this season.")


            elif subitem['type'] == 'team_selector':
                selector_col1, selector_col2 = st.columns(2)
                with selector_col1:
                    team = st.selectbox(subitem['label'], df['team'].unique(), key=subitem['key'])
                with selector_col2:
                    opponent = st.selectbox("Select Opponent", df['team'].unique(), key=f"{subitem['key']}_opponent")

                if team and opponent and model_path:
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.subheader(f"Current ability of {team}")
                        max_season = df[df['team'] == team]['season'].max()
                        team_league = df[(df['team'] == team) & (df['season'] == max_season)]['league_name'].iloc[0]
                        league_teams = df[(df['league_name'] == team_league) & (df['season'] == max_season)]['team'].unique()
                        
                        theta_mean = xr.open_dataarray(os.path.join(model_path, "bayesian_theta_mean.nc"))
                        
                        season_abilities = theta_mean.sel(season=max_season)
                        
                        valid_league_teams = [t for t in league_teams if t in season_abilities.team.values]
                        league_team_abilities = season_abilities.sel(team=valid_league_teams)
                        
                        league_team_abilities_pd = league_team_abilities.to_pandas().sort_values(ascending=False)

                        fig = go.Figure()
                        colors = ['red' if t == team else 'blue' for t in league_team_abilities_pd.index]
                        fig.add_trace(go.Bar(
                            x=league_team_abilities_pd.index,
                            y=league_team_abilities_pd.values,
                            marker_color=colors
                        ))
                        fig.update_layout(title=f'Team Ability in {team_league} ({max_season})', xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                    with chart_col2:
                        st.subheader(f"Effective Stats against {opponent} (Ranked highest to lowest on Win Probability)")
                        opponent_games = fixture_dat_with_preds[fixture_dat_with_preds['opponent'] == opponent]
                        if not opponent_games.empty:
                            cluster_effectiveness = opponent_games.groupby('cluster_rank')['predicted_win_prob_mean'].mean().reset_index()
                            cluster_info = pd.merge(cluster_effectiveness, clusters, on='cluster_rank')
                            cluster_info_sorted = cluster_info.sort_values('predicted_win_prob_mean', ascending=True)
                            top_10_clusters = cluster_info_sorted.head(10)
                            st.dataframe(top_10_clusters)
                        else:
                            st.write("No data available for this opponent.")

                    # --- New Plot: Team Ability Distribution ---
                    st.subheader("Team Ability Distribution Comparison")
                    
                    # Load required data
                    theta_mean = xr.open_dataarray(os.path.join(model_path, "bayesian_theta_mean.nc"))
                    theta_sd = xr.open_dataarray(os.path.join(model_path, "bayesian_theta_sd.nc"))

                    # Check which of the selected teams are available in the model data for the season
                    season_abilities = theta_mean.sel(season=max_season)
                    available_teams_in_model = season_abilities.team.values
                    
                    # Get unique list of teams that are available in the model
                    selected_teams = [team, opponent]
                    teams_to_plot = list(set([t for t in selected_teams if t in available_teams_in_model]))
                    
                    if teams_to_plot:
                        # Calculate league average
                        league_teams = df[(df['league_name'] == team_league) & (df['season'] == max_season)]['team'].unique()
                        valid_league_teams = [t for t in league_teams if t in season_abilities.team.values]
                        league_team_abilities = season_abilities.sel(team=valid_league_teams)
                        league_average_ability = league_team_abilities.mean().item()

                        # Get mu and sigma for the available teams
                        mu = theta_mean.sel(team=teams_to_plot, season=max_season).to_pandas()
                        sigma = theta_sd.sel(team=teams_to_plot, season=max_season).to_pandas()

                        # Generate posterior samples for each available team
                        post_dfs = []
                        for t in teams_to_plot:
                            df_dist = pd.DataFrame({
                                "team": [t] * 1000,
                                "dist": sc.norm(mu[t], sigma[t]).rvs(1000)
                            })
                            post_dfs.append(df_dist)
                        
                        post_df = pd.concat(post_dfs, axis=0)

                        # Create the violin plot with Plotly
                        fig = go.Figure()

                        for t in teams_to_plot:
                            fig.add_trace(go.Violin(x=post_df[post_df['team'] == t]['dist'],
                                                   name=t,
                                                   box_visible=True,
                                                   meanline_visible=True))
                        
                        title_teams = " vs ".join(teams_to_plot)
                        fig.update_layout(
                            title_text=f'Posterior Ability Distribution: {title_teams}',
                            xaxis_title='Team Ability',
                            yaxis_title='Density',
                            showlegend=True
                        )

                        fig.add_shape(type='line',
                                      x0=league_average_ability, y0=0, x1=league_average_ability, y1=1,
                                      yref='paper',
                                      line=dict(color='grey', dash='dash'),
                                      name='League Average')
                        
                        st.plotly_chart(fig)
                    
                    # Notify user if some teams were not found
                    missing_teams = set(selected_teams) - set(teams_to_plot)
                    if missing_teams:
                        st.write(f"Note: Ability distribution data not available for {', '.join(missing_teams)} in season {max_season}.")

                    # --- Section: Cluster Ranks ---
                    st.subheader("Recent Cluster Ranks")

                    # Head-to-head matches
                    st.write(f"**Last 5 Head-to-Head Matches between {team} and {opponent}:**")
                    head_to_head_games = fixture_dat_with_preds[
                        (fixture_dat_with_preds['team'] == team) & (fixture_dat_with_preds['opponent'] == opponent)
                    ].copy()

                    if not head_to_head_games.empty:
                        if 'fixture_date' in head_to_head_games.columns:
                            head_to_head_games['fixture_date'] = pd.to_datetime(head_to_head_games['fixture_date'])
                            latest_games = head_to_head_games.sort_values('fixture_date', ascending=False).head(5)
                            
                            display_df = latest_games[['fixture_date', 'cluster_rank', 'cluster_rank_opponent_km']].copy()
                            display_df.rename(columns={
                                'fixture_date': 'Date',
                                'cluster_rank': f'{team} Cluster Rank',
                                'cluster_rank_opponent_km': f'{opponent} Cluster Rank'
                            }, inplace=True)
                            
                            st.dataframe(display_df)
                        else:
                            st.write("Date information ('fixture_date') not found to show recent games.")
                    else:
                        st.write(f"No head-to-head games found between {team} and {opponent} in the dataset.")

                    # General recent matches for each team
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Last 5 Matches for {team}:**")
                        team_games = fixture_dat_with_preds[fixture_dat_with_preds['team'] == team].copy()
                        if not team_games.empty:
                            if 'fixture_date' in team_games.columns:
                                team_games['fixture_date'] = pd.to_datetime(team_games['fixture_date'])
                                latest_team_games = team_games.sort_values('fixture_date', ascending=False).head(5)
                                
                                display_df_team = latest_team_games[['fixture_date', 'opponent', 'cluster_rank', 'cluster_rank_opponent_km']].copy()
                                display_df_team.rename(columns={
                                    'fixture_date': 'Date',
                                    'opponent': 'Opponent',
                                    'cluster_rank': 'Cluster Rank',
                                    'cluster_rank_opponent_km': 'Opponent Cluster Rank'
                                }, inplace=True)
                                st.dataframe(display_df_team)
                            else:
                                st.write("Date information ('fixture_date') not found.")
                        else:
                            st.write(f"No matches found for {team}.")

                    with col2:
                        st.write(f"**Last 5 Matches for {opponent}:**")
                        opponent_games_df = fixture_dat_with_preds[fixture_dat_with_preds['team'] == opponent].copy()
                        if not opponent_games_df.empty:
                            if 'fixture_date' in opponent_games_df.columns:
                                opponent_games_df['fixture_date'] = pd.to_datetime(opponent_games_df['fixture_date'])
                                latest_opponent_games = opponent_games_df.sort_values('fixture_date', ascending=False).head(5)
                                
                                display_df_opponent = latest_opponent_games[['fixture_date', 'opponent', 'cluster_rank', 'cluster_rank_opponent_km']].copy()
                                display_df_opponent.rename(columns={
                                    'fixture_date': 'Date',
                                    'opponent': 'Opponent',
                                    'cluster_rank': 'Cluster Rank',
                                    'cluster_rank_opponent_km': 'Opponent Cluster Rank'
                                }, inplace=True)
                                st.dataframe(display_df_opponent)
                            else:
                                st.write("Date information ('fixture_date') not found.")
                        else:
                            st.write(f"No matches found for {opponent}.")


            elif subitem['type'] == 'variable_selector':
                variables = st.multiselect(subitem['label'], df.select_dtypes(include=np.number).columns, key=subitem['key'])
                if variables:
                    corr_df = df[variables]
                    fig = plot_correlation_matrix(corr_df)
                    st.plotly_chart(fig)

            