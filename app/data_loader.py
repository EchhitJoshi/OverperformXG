import pandas as pd
import numpy as np
import yaml
import requests
import time
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
    
    #team_fixtures[['fixture_id','teams_home_name','teams_away_name']].to_parquet(home_dir + f"/data/Fixtures/{team}_{season}_{league}.parquet")
    return team_fixtures#[['fixture_id','teams_home_name','teams_away_name']]
    
    
    
        
def read_fixtures_for_season(team,season):
    """
    team: Name of team
    season: int, year of the season start
    Returns data for the season for the specified team with some engineered features
    """
    fixtures = get_team_fixtures(team,season)
    fixtures['winner'] = np.where(fixtures.teams_home_winner == True,fixtures.teams_home_name,
                                   np.where(fixtures.teams_away_name == True,fixtures.teams_away_name,'Draw'))
    home_fixtures = list(fixtures[fixtures.teams_home_name == team]['fixture_id'])
    away_fixtures = list(fixtures[fixtures.teams_away_name == team]['fixture_id'])

    fixtures_dat = pd.DataFrame()

    for fixture in home_fixtures + away_fixtures:
        player_stat_url = "https://v3.football.api-sports.io/fixtures/players?fixture={}".format(fixture)
        fixture_dat = requests.get(player_stat_url,headers=headers_api_sport)
        time.sleep(15)
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
            

        fixtures_dat = pd.concat([fixtures_dat,fixture_dat_expanded],axis = 0)
        team_winner = str(fixtures[(fixtures.fixture_id == fixture)]['winner'].values[0])
        fixtures_dat['team_winner'] = team_winner
        fixtures_dat['team'] = team

    # Outcome
    fixtures_dat['outcome'] = np.where(fixtures_dat.team == fixtures_dat.team_winner,'win',np.where(fixtures_dat.team_winner == 'Draw','draw','loss'))
    
    # feature engineering
    fixtures_dat['dribble_success_rate'] = (fixtures_dat.dribbles_success/fixtures_dat.dribbles_attempts) * 100
    fixtures_dat['target_shot_cr'] = (fixtures_dat.goals_total/fixtures_dat.shots_on) * 100

    fixtures_dat = lower_columns(fixtures_dat)

    return fixtures_dat




    
        
    

    
