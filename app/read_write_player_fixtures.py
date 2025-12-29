import os
import time
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import argparse

from data_loader import get_leagues, read_fixtures_for_season, create_datetime_columns
from utils import home_dir, get_season

# Config
# Assuming the script is run from the project root.
config_path = 'app/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
# Mysql string
db_url = config['MYSQL_STRING']


def write_df_to_db(df, table_name, db_type="postgres", db_url=None, if_exists="append", chunksize=5000):
    """
    Write a pandas DataFrame to either PostgreSQL or MySQL based on db_type.
    
    Args:
        df (pd.DataFrame): DataFrame to write
        table_name (str): Table name
        db_type (str): "postgres" or "mysql"
        db_url (str): Full SQLAlchemy DB URL
        if_exists (str): 'append', 'replace', or 'fail'
        chunksize (int): Number of rows per batch
    """
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    if db_url is None:
        raise ValueError("db_url must be provided")

    # Convert unsupported types
    for col in df.columns:
        if pd.api.types.is_period_dtype(df[col]):
            df[col] = df[col].dt.to_timestamp()
        elif pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Ensure the driver is correct
    if db_type.lower() == "postgres":
        # PostgreSQL usually uses psycopg2
        if "postgresql" not in db_url:
            db_url = "postgresql+psycopg2://" + db_url.split("://")[1]
    elif db_type.lower() == "mysql":
        # MySQL usually uses pymysql
        if "mysql" not in db_url:
            db_url = "mysql+pymysql://" + db_url.split("://")[1]
    else:
        raise ValueError("db_type must be 'postgres' or 'mysql'")

    # Create engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    # Write using session manager
    with Session() as session:
        try:
            df.to_sql(
                table_name,
                con=session.connection(),
                if_exists=if_exists,
                index=False,
                chunksize=chunksize
            )
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error writing to table '{table_name}': {e}")
            raise

# Load reference data once
# The home_dir variable is expected to be defined in utils.py and point to the project root.
leagues_dat = pd.read_sql("select * from overperformxg.league",config['MYSQL_STRING'])
teams_data = pd.read_sql("select * from overperformxg.team_league_map",config['MYSQL_STRING'])
teams_data = teams_data.merge(leagues_dat, left_on='league', right_on="league_id", how="left")


def update_team_season_fixtures(team_name, season):
    """
    Checks for updates for a given team and season, fetches data if needed,
    and writes to the database.
    """
    engine = create_engine(db_url)


    
    max_fixture_date = None
    try:
        query = f"SELECT MAX(fixture_date) as max_date FROM overperformxg.complete_data WHERE team = '{team_name}' AND season = {season}"
        result = pd.read_sql(query, engine)
        if not result.empty and result['max_date'].iloc[0] is not None:
            max_fixture_date = pd.to_datetime(result['max_date'].iloc[0])
    except Exception as e:
        print(f"Could not fetch max fixture date from DB for team {team_name} (ID: {team_id}), season {season}. Error: {e}")

    if max_fixture_date is None or datetime.now() > max_fixture_date:
        print(f"Checking for new fixture data for team {team_name}, season {season}.")
        
        # This function is expected to fetch data and save a parquet file. It needs team name.
        read_fixtures_for_season(team_name, season, sleep_time=20)

        fixtures_dir = os.path.join(home_dir, "data/Fixtures")
        fixture_file = os.path.join(fixtures_dir, f"{team_name.replace(' ', '_')}_{season}.parquet")

        if os.path.exists(fixture_file):
            print(f"Processing fixture file: {fixture_file}")
            df = pd.read_parquet(fixture_file)
            
            if df.empty:
                print(f"Fixture file for team {team_name}, season {season} is empty. Nothing to process.")
                return

            df = df.reset_index(drop=True)

            # Data checks and transformations
            if 'passes_accuracy' in df.columns:
                df['passes_accuracy'] = df['passes_accuracy'].astype("float64")
                df.rename(columns={'passes_accuracy': 'passes_accurate'}, inplace=True)
            
            df['fixture_date'] = pd.to_datetime(df['fixture_date'])
            df['fixture_date_dt'] = df['fixture_date'].dt.date
            df = create_datetime_columns(df, 'fixture_date')
            
            if 'games_rating' in df.columns:
                df['games_rating'] = pd.to_numeric(df['games_rating'], errors='coerce')

            df['season'] = df['fixture_date'].apply(get_season)

            # Targets
            if 'outcome' in df.columns:
                df['outcome_num'] = pd.Categorical(df.outcome).codes
                df['win'] = np.where(df.outcome.str.lower() == 'win', 1, 0)
                df['draw'] = np.where(df.outcome.str.lower() == 'draw', 1, 0)
                df['loss'] = np.where(df.outcome.str.lower() == 'loss', 1, 0)

            # Joins
            df = df.merge(
                teams_data[['team_id', 'team_name', 'league_id', 'league_name']].drop_duplicates(subset=['team_id']),
                how='left',
                on='team_id'
            )
            df = df.drop_duplicates()

            # Drop duplicates already in the database
            try:
                existing_fixture_ids_query = f"SELECT DISTINCT fixture_id FROM overperformxg.complete_data WHERE team_name = '{team_name}' AND season = {season}"
                fixtures_id_df = pd.read_sql(existing_fixture_ids_query, con=engine)
                if not fixtures_id_df.empty:
                    existing_fixture_ids = fixtures_id_df['fixture_id'].tolist()
                    df = df[~df.fixture_id.isin(existing_fixture_ids)]
            except Exception as e:
                print(f"Could not fetch existing fixture IDs from DB. Writing all data. Error: {e}")

            # Write to DB
            if not df.empty:
                print(f"Writing {len(df)} new records to the database.")
                write_df_to_db(df, "complete_data", db_type="mysql", db_url=db_url)
            else:
                print("No new records to write to the database.")
        else:
            print(f"Fixture file for team {team_name}, season {season} not found after attempting to fetch.")
    else:
        print(f"Fixture data for team {team_name}, season {season} is up-to-date (last fixture on {max_fixture_date.date()}).")


def main():
    parser = argparse.ArgumentParser(description="Fetch and update fixture data for a specific team and season.")
    parser.add_argument("--team_name", type=str, required=True, help="Team name to process.")
    parser.add_argument("--season", type=int, required=True, help="Season to process (e.g., 2023 for 2023/2024 season).")
    args = parser.parse_args()
    
    update_team_season_fixtures(args.team_name, args.season)


if __name__ == "__main__":
    main()
    print("Script finished.")