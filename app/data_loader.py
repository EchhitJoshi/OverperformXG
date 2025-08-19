import pandas as pd
import numpy as np
import yaml
import requests
import time
import re
#from sqlalchemy import create_engine

# Configs
with open("config.yaml",'r') as f:
    config = yaml.safe_load(f)
home_dir = config['HOME_DIRECTORY']
headers_api_sport = config["HEADERS_API_SPORT"]

def get_mysql_con():
    """
    returns the mysql connection engine
    """
    return create_engine("mysql+pymysql://echhitjoshi:mz4DTyW6iyiJEnzdnmRg@database-eonjive.ctkiqcsu6x6o.us-east-1.rds.amazonaws.com")

def lower_columns(df):
    df.columns = df.columns.str.replace(r"[ .-]","_",regex = True)
    return df

def read_sql(query,con):
    """
    query: sql query string
    con: connection engine
    returns extracted data with columns lowered and joined with '_'
    """
    with con.connect() as conn:
        raw_dat =  pd.read_sql(query,con = conn)
    
    raw_dat = lower_columns(raw_dat)
    print(raw_dat.info())
    return raw_dat
    
def create_datetime_columns(data,dt_col):
    """
    data: pandas DF
    dt_col: datetime column to parse dt values
    returns year, month(floored month/year), month_name, day_of_week, day_of_week_name, week( floored week)
    suffixed with _e for uniqueness
    """
    
    data[dt_col] = pd.to_datetime(data[dt_col])
    data['year_e'] = data[dt_col].dt.year
    data['month_e'] = data[dt_col].dt.to_period('M')
    data['month_name_e'] = data[dt_col].dt.month_name()
    data['day_of_week_e'] = data[dt_col].dt.day_of_week
    data['day_of_week_name_e'] = data[dt_col].dt.day_name()
    data['week_e'] = data[dt_col].dt.to_period('W-MON')
    data['week_e'] = data['week_e'].dt.start_time
    return data


def get_leagues(leagues_path = home_dir + "/data/Leagues/leagues.parquet"):
    return lower_columns(pd.read_parquet(leagues_path))

        
def get_team_fixtures(team, season, league = None):
    if not league:
        # read team_league_map
        team_league_map = pd.read_parquet(home_dir + "/data/Teams/team_league.parquet")
        league = team_league_map[team_league_map.team_name == team]['league'].unique()[0]
    
    print(f"Pulling for {team},{season} with league_id: {league}")
    url = "https://v3.football.api-sports.io/fixtures?league={}&season={}".format(league,season)
    fixtures_response = requests.get(url,headers=headers_api_sport)
    fixtures_dat = pd.json_normalize(fixtures_response.json()['response'])
    fixtures_dat = lower_columns(fixtures_dat)
    team_fixtures = fixtures_dat[(fixtures_dat.teams_home_name.str.lower().str.contains(team.lower()) ) | (fixtures_dat.teams_away_name.str.lower().str.contains(team.lower()))]
    
    team_fixtures['winner'] = np.where(team_fixtures.teams_home_winner == True,team_fixtures.teams_home_name,
                                   np.where(team_fixtures.teams_away_winner == True,team_fixtures.teams_away_name,'Draw'))

    team_fixtures.to_parquet(home_dir + f"/data/All_Fixtures/{team}_{season}.parquet")
    #team_fixtures[['fixture_id','teams_home_name','teams_away_name']].to_parquet(home_dir + f"/data/Fixtures/{team}_{season}_{league}.parquet")
    return team_fixtures#[['fixture_id','teams_home_name','teams_away_name']]
    
    





        
def read_fixtures_for_season(team,season):
    """
    team: Name of team
    season: int, year of the season start
    Returns data for the season for the specified team with some engineered features
    """
    print(f"processing for {team}, {season}")
    fixtures = get_team_fixtures(team,season)
    
    
    
    home_fixtures = list(fixtures[fixtures.teams_home_name == team]['fixture_id'])
    away_fixtures = list(fixtures[fixtures.teams_away_name == team]['fixture_id'])

    fixtures_dat = pd.DataFrame()

    for fixture in home_fixtures + away_fixtures:
        player_stat_url = "https://v3.football.api-sports.io/fixtures/players?fixture={}".format(fixture)
        fixture_dat = requests.get(player_stat_url,headers=headers_api_sport)
        time.sleep(2)
        if fixture in home_fixtures:
            #process for home
            fixture_dat_expanded = pd.concat([pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])[['player.id','player.name']],pd.json_normalize(pd.json_normalize(pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])['statistics']).rename(columns = {0:"player_stats"})['player_stats'])],axis = 1)
            fixture_dat_expanded['fixture_id'] = fixture
            fixture_dat_expanded['team_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_home'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_scored_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_home'].values[0] 
            fixture_dat_expanded['team_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_away'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_conceded_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_away'].values[0]             
            fixture_dat_expanded['opponent'] = fixtures[(fixtures.fixture_id == fixture)]['teams_away_name'].values[0]             
            
        else:
            #process for away
            fixture_dat_expanded = pd.concat([pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][1])[['player.id','player.name']],pd.json_normalize(pd.json_normalize(pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][1])['statistics']).rename(columns = {0:"player_stats"})['player_stats'])],axis = 1)
            fixture_dat_expanded['fixture_id'] = fixture
            fixture_dat_expanded['team_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_away'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_scored_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_away'].values[0] 
            fixture_dat_expanded['team_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_home'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_conceded_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_home'].values[0] 
            fixture_dat_expanded['opponent'] = fixtures[(fixtures.fixture_id == fixture)]['teams_home_name'].values[0]   

        # adding team winner
        fixture_dat_expanded['fixture_date'] = fixtures[(fixtures.fixture_id == fixture)]['fixture_date'].values[0] 
        fixture_dat_expanded['team_winner'] = str(fixtures[(fixtures.fixture_id == fixture)]['winner'].values[0])

        fixtures_dat = pd.concat([fixtures_dat,fixture_dat_expanded],axis = 0)
        fixtures_dat['team'] = team
        
        
    
    fixtures_dat = lower_columns(fixtures_dat)

    # Outcome
    fixtures_dat['outcome'] = np.where(fixtures_dat.team == fixtures_dat.team_winner,'win',np.where(fixtures_dat.team_winner == 'Draw','draw','loss'))

    # feature engineering
    fixtures_dat['dribble_success_rate'] = (fixtures_dat.dribbles_success.astype("float64")/fixtures_dat.dribbles_attempts.astype("float64")) * 100
    fixtures_dat['target_shot_conversion_perc'] = (fixtures_dat.goals_total.astype("float64")/fixtures_dat.shots_on.astype("float64")) * 100
    fixtures_dat['duels_won_perc'] = (fixtures_dat.duels_won.astype("float64")/fixtures_dat.duels_total.astype("float64")) * 100
    fixtures_dat['pass_accuracy_perc'] = (fixtures_dat.passes_accuracy.astype("float64")/ fixtures_dat.passes_total.astype("float64")) * 100

    fixtures_dat.to_parquet(home_dir+f"/data/Fixtures/{team.replace(' ','_')}_{str(season)}.parquet")

    return fixtures_dat






def read_fixture_stats(dat,team,season,missing_features:list):
    '''
    team: team to add missing fixture features
    season: season in question
    missing_features: features to add from fixtures data
    '''
    fixtures = get_team_fixtures(team,season)    
    home_fixtures = list(fixtures[fixtures.teams_home_name == team]['fixture_id'])
    away_fixtures = list(fixtures[fixtures.teams_away_name == team]['fixture_id'])

    fixtures.to_parquet(home_dir + f"/data/All_Fixtures/{team}_{season}.parquet")
    
    fixture_dat = pd.DataFrame()
    for fixture in home_fixtures + away_fixtures:

        if fixture in home_fixtures:
            fixture_dat_expanded = pd.concat([pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])[['player.id','player.name']],pd.json_normalize(pd.json_normalize(pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][0])['statistics']).rename(columns = {0:"player_stats"})['player_stats'])],axis = 1)
            fixture_dat_expanded['fixture_id'] = fixture
            fixture_dat_expanded['team_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_home'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_scored_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_home'].values[0] 
            fixture_dat_expanded['team_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_away'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_conceded_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_away'].values[0]             
            fixture_dat_expanded['opponent'] = fixtures[(fixtures.fixture_id == fixture)]['teams_away_name'].values[0]             
        else:
            fixture_dat_expanded = pd.concat([pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][1])[['player.id','player.name']],pd.json_normalize(pd.json_normalize(pd.json_normalize(pd.json_normalize(fixture_dat.json()['response'])['players'][1])['statistics']).rename(columns = {0:"player_stats"})['player_stats'])],axis = 1)
            fixture_dat_expanded['fixture_id'] = fixture
            fixture_dat_expanded['team_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_scored'] = fixtures[(fixtures.fixture_id == fixture)]['goals_away'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_away'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_scored_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_away'].values[0] 
            fixture_dat_expanded['team_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] 
            fixture_dat_expanded['team_non_penalty_goals_conceded'] = fixtures[(fixtures.fixture_id == fixture)]['goals_home'].values[0] - fixtures[(fixtures.fixture_id == fixture)]['score_penalty_home'].fillna(0).values[0]
            fixture_dat_expanded['team_goals_conceded_half'] = fixtures[(fixtures.fixture_id == fixture)]['score_halftime_home'].values[0] 
            fixture_dat_expanded['opponent'] = fixtures[(fixtures.fixture_id == fixture)]['teams_home_name'].values[0]   


    return pd.merge(dat,fixture_dat, on = 'fixture_id',how = 'left')

    



def combine_fixture_stats(complete_data):

    fixture_df = complete_data.groupby(['fixture_id','team']).agg(total_rating = ("games_rating","sum"),
                                                                  total_games_played_by_player = ("games_number","sum"),
                                                                  total_shots = ("shots_total","sum"),
                                                                  total_shots_on_target = ("shots_on","sum"),
                                                                  total_goals = ("goals_total","sum"),
                                                                  total_goals_conceded = ("goals_conceded","sum"),
                                                                  total_goals_assists = ("goals_assists","sum"),
                                                                  total_goals_saves = ("goals_saves","sum"),
                                                                  total_passes = ("passes_total","sum"),
                                                                  total_passes_key = ("passes_key","sum"),
                                                                  total_tackles = ("tackles_total","sum"),
                                                                  total_tackles_blocks = ("tackles_blocks","sum"),
                                                                  total_tackles_interceptions = ("tackles_interceptions","sum"),
                                                                  total_duels = ("duels_total","sum"),
                                                                  total_duels_won = ("duels_won","sum"),
                                                                  total_dribbles_attempts = ("dribbles_attempts","sum"),
                                                                  total_dribbles_success = ("dribbles_success","sum"),
                                                                  total_dribbles_past = ("dribbles_past","sum"),
                                                                  total_fouls_drawn = ("fouls_drawn","sum"),
                                                                  
                                                                  total_fouls_committed = ("fouls_committed","sum"),
                                                                  total_cards_yellow = ("cards_yellow","sum"),
                                                                  total_cards_red = ("cards_red","sum"),
                                                                  total_penalty_won = ("penalty_won","sum"),
                                                                  total_penalty_commited = ("penalty_commited","sum"),
                                                                  total_penalty_scored = ("penalty_scored","sum"),
                                                                  total_penalty_saved = ("penalty_saved","sum"),
                                                                #  total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),
                                                                #   total_fouls_drawn = ("fouls_drawn","sum"),

                                                                  






                                                                  avg_rating = ("games_rating","mean"),
                                                                  avg_shots = ("shots_total","mean"),
                                                                  avg_pass_accuracy = ("passes_accuracy","mean"),

                                                                  #avg_key_passes_per_midfield = ("passes_key_passes",) later







                                                                  )
    

    
def compare_players(dat,players:list,years:list,transpose:str):
    
    final_dat = dat[(dat.player_name.isin(players)) & (dat.year_e.isin(years)) ].groupby(["player_name","year_e"],as_index = False).apply(lambda group: pd.Series({
    "player_id": group["player_id"].iloc[0],
    "player_name": group["player_name"].iloc[0],
    "total_games_played": group["fixture_id"].count(),
    "total_minutes_played": group["games_minutes"].sum(),
    "average_rating": group["games_rating"].mean(),
    "captain_matches": group["games_captain"].sum(),
    "substitute_appearances": group["games_substitute"].sum(),
    "total_shots": group["shots_total"].sum(),
    "shots_on_target": group["shots_on"].sum(),
    "goals_scored": group["goals_total"].sum(),
    "assists": group["goals_assists"].sum(),
    "yellow_cards": group["cards_yellow"].sum(),
    "red_cards": group["cards_red"].sum(),
    "fouls_drawn": group["fouls_drawn"].sum(),
    "fouls_committed": group["fouls_committed"].sum(),
    "attempted_dribbles": group["dribbles_attempts"].sum(),
    "successful_dribbles": group["dribbles_success"].sum(),
    "dribbled_past": group["dribbles_past"].sum(),
    "dribble_success_rate": group["dribbles_success"].sum() / group["dribbles_attempts"].sum() if group["dribbles_attempts"].sum() > 0 else 0,
    "total_passes": group["passes_total"].sum(),
    "key_passes": group["passes_key"].sum(),
    "average_passes_accurate": group["passes_accurate"].mean(),
    "average_pass_accuracy": group["pass_accuracy_perc"].mean(),
    "total_tackles": group["tackles_total"].sum(),
    "blocks": group["tackles_blocks"].sum(),
    "interceptions": group["tackles_interceptions"].sum(),
    "duels_contested": group["duels_total"].sum(),
    "duels_won": group["duels_won"].sum(),
    "duels_won_percentage": group["duels_won"].sum() / group["duels_total"].sum() if group["duels_total"].sum() > 0 else 0,
    "penalties_won": group["penalty_won"].sum(),
    "penalties_committed": group["penalty_commited"].sum(),
    "penalties_scored": group["penalty_scored"].sum(),
    "penalties_missed": group["penalty_missed"].sum(),
    "penalties_saved": group["penalty_saved"].sum(),
    "team_goals_scored": group["team_goals_scored"].sum(),
    "team_non_penalty_goals": group["team_non_penalty_goals_scored"].sum(),
    "team_goals_conceded": group["team_goals_conceded"].sum(),
    "team_non_penalty_goals_conceded": group["team_non_penalty_goals_conceded"].sum(),
    "matches_won": group["win"].sum(),
    # Per 90 stats
    "total_shots_per_90": group["shots_total"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "shots_on_target_per_90": group["shots_on"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "goals_scored_per_90": group["goals_total"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "assists_per_90": group["goals_assists"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "yellow_cards_per_90": group["cards_yellow"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "red_cards_per_90": group["cards_red"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "fouls_drawn_per_90": group["fouls_drawn"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "fouls_committed_per_90": group["fouls_committed"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "attempted_dribbles_per_90": group["dribbles_attempts"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "successful_dribbles_per_90": group["dribbles_success"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "dribbled_past_per_90": group["dribbles_past"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "dribble_success_rate_per_90": (group["dribbles_success"].sum() / group["dribbles_attempts"].sum() if group["dribbles_attempts"].sum() > 0 else 0) * 90,
    "total_passes_per_90": group["passes_total"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "key_passes_per_90": group["passes_key"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "average_passes_accurate_per_90": group["passes_accurate"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "average_pass_accuracy_per_90": (group["pass_accuracy_perc"] * group["games_minutes"]).sum()/group["games_minutes"].sum(),
    "total_tackles_per_90": group["tackles_total"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "blocks_per_90": group["tackles_blocks"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "interceptions_per_90": group["tackles_interceptions"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "duels_contested_per_90": group["duels_total"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "duels_won_per_90": group["duels_won"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "duels_won_percentage_per_90": (group["duels_won"].sum() / group["duels_total"].sum() if group["duels_total"].sum() > 0 else 0) * 90,
    "penalties_won_per_90": group["penalty_won"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "penalties_committed_per_90": group["penalty_commited"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "penalties_scored_per_90": group["penalty_scored"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "penalties_missed_per_90": group["penalty_missed"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    "penalties_saved_per_90": group["penalty_saved"].sum() / group["games_minutes"].sum() * 90 if group["games_minutes"].sum() > 0 else 0,
    }))

    if transpose:
        final_dat = final_dat.T
        final_dat.columns = final_dat.loc['player_name'].values
        final_dat.drop(index = 'player_name',inplace = True)
        final_dat.index.name = 'stat_type'
    else:
        pass

    return final_dat


def calculate_fixture_stats(dat):
    aggregations = {
        'offsides': [ 'mean'],
        'games_minutes': [ 'mean'],
        'games_number': ['mean'],
        'games_rating': ['mean'],
        'games_captain': [ 'mean'],
        'games_substitute': [ 'mean'],
        'shots_total': ['sum'],
        'shots_on': ['sum'],
        'goals_total': ['sum'],
        'goals_assists': ['sum'],
        'goals_saves': ['sum'],
        'passes_total': ['sum'],
        'passes_key': ['sum'],
        'passes_accurate': ['sum'],
        'tackles_total': ['sum'],
        'tackles_blocks': ['sum'],
        'tackles_interceptions': ['sum'],
        'duels_total': ['sum'],
        'duels_won': ['sum'],
        'dribbles_attempts': ['sum'],
        'dribbles_success': ['sum'],
        'dribbles_past': ['sum'],
        'fouls_drawn': ['sum'],
        'fouls_committed': ['sum'],
        'cards_yellow': ['sum'],
        'cards_red': ['sum'],
        'penalty_won': ['sum'],
        'penalty_commited': ['sum'],
        'penalty_scored': ['sum'],
        'penalty_missed': ['sum'],
        'penalty_saved': ['sum'],
        'dribble_success_rate': ['mean'],
        'target_shot_conversion_perc': ['mean'],
        'duels_won_perc': ['mean'],
        'pass_accuracy_perc': ['mean'],
        'win':['unique']
    }

    # Perform the groupby and aggregation
    aggregated = dat[dat.games_position.isin(["M","F","D"])].groupby(['fixture_id', 'team', 'games_position']).agg(aggregations)

    # Flatten column names to remove multi-index for aggregated statistics
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]


    # Reset index for pivoting/grouped analysis
    aggregated = aggregated.reset_index()


    #print(aggregated.head())

    # Filter relevant columns for the fixture level (team-wide stats, no aggregation needed)
    team_level_columns = [
        'fixture_id', 'team', 'team_goals_scored', 'team_non_penalty_goals_scored',
        'team_goals_scored_half', 'team_goals_conceded',
        'team_non_penalty_goals_conceded', 'team_goals_conceded_half',
        'team_winner', 'opponent'
    ]

    # Extract team-wide stats (no aggregation needed)
    team_data = dat[team_level_columns].drop_duplicates()


    # Merge the team-wide stats back with the position-based aggregated stats
    merged_data = aggregated.merge(team_data, on=['fixture_id', 'team'], how='left')

    # Pivot the data to have one row per fixture_id + team and columns as aggregated stats per games_position
    # result = merged_data.pivot_table(
    #     index=['fixture_id', 'team'],  # Group by fixture_id and team
    #     columns='games_position',     # Separate columns by games_position
    #     values=[col for col in merged_data.columns if col not in ['fixture_id', 'team', 'games_position']],
    #     aggfunc='first'               # Only one value per attribute for each position
    # )

    # # Flatten the resulting pivot table's columns
    # result.columns = ['_'.join([str(c) for c in col]).strip() for col in result.columns]

    # Reset index to have fixture_id and team as columns
    #result = result.reset_index()

    # Final wrangling
    merged_data = merged_data.fillna(0)
    merged_data.columns = [re.sub(r"(_sum|_mean|_unique)$","",col) for col in merged_data.columns]    
    merged_data['win'] = merged_data['win'].apply(lambda x: x[0])

    # Add dates:
    fixture_date = dat[['fixture_id','week_e','year_e']].drop_duplicates()
    merged_data = pd.merge(merged_data,fixture_date[['fixture_id','week_e','year_e']],how = 'left',on = 'fixture_id')

    return merged_data

    # --- End of generated code block ---

    
