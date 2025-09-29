
import streamlit as st
import yaml
import sys
import os
import pandas as pd
import numpy as np

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from data_loader import *
from visualizations import *
from utils import * 
from models import *

# Load configuration
with open("streamlit/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

with open(os.path.join(os.path.dirname(__file__), '..', 'app', 'config.yaml'), 'r') as f:
    app_config = yaml.safe_load(f)
home_dir = app_config['HOME_DIRECTORY']

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    # The data loading logic from main.py
    fixtures_dir = os.path.join(home_dir, "data", "Fixtures")

    leagues_dat = get_leagues(home_dir +"/data/Leagues/leagues.parquet")
    teams_dat = pd.read_parquet(home_dir + "/data/Teams/team_league.parquet")

    complete_data = pd.DataFrame()
    for file in os.listdir(fixtures_dir):
        dat = pd.read_parquet(os.path.join(fixtures_dir,file))
        complete_data = pd.concat([complete_data,dat],axis = 0)

    complete_data = complete_data.reset_index()
    complete_data.drop(columns = ['index'],inplace=True)

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

    # Get major position
    player_position_map = get_major_position(complete_data)
    complete_data = complete_data.merge(player_position_map, on='player_id', how='left')

    # Create player stats df
    player_stats_df = compare_players(complete_data, complete_data['player_name'].unique(), complete_data['season'].unique(), transpose=False)
    player_stats_df = player_stats_df.merge(player_position_map, on='player_id', how='left')

    # Calculate percentile ranks
    per_90_stats = [col for col in player_stats_df.columns if '_per_90' in col]
    for stat in per_90_stats:
        player_stats_df[f'{stat}_percentile'] = player_stats_df.groupby(['major_position', 'season'])[stat].rank(pct=True) * 100

    # Get player clusters
    player_stats_df = train_kmeans_and_get_clusters(player_stats_df, n_clusters=50)

    return complete_data, player_stats_df

df, player_stats_df = load_data()

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
                teams = st.selectbox(subitem['label'], df['team'].unique(), key=subitem['key'])
                if teams:
                    team_data = df[df['team'] == teams]
                    fig = plot_team_performance(team_data, teams)
                    st.plotly_chart(fig)

            elif subitem['type'] == 'variable_selector':
                variables = st.multiselect(subitem['label'], df.select_dtypes(include=np.number).columns, key=subitem['key'])
                if variables:
                    corr_df = df[variables]
                    fig = plot_correlation_matrix(corr_df)
                    st.plotly_chart(fig)

            elif subitem['type'] == 'chart':
                st.subheader(subitem['title'])
                # Placeholder for chart
                st.write(f"Chart type: {subitem['chart_type']}")
