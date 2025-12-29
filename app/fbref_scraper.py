import httpx
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# List of common user agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0',
]

def get_random_user_agent():
    """Returns a random user agent from the list."""
    return random.choice(USER_AGENTS)

def scrape_fixture_player_stats(fixture_url):
    """
    Scrapes player stats for a given fixture from FBRef.

    Args:
        fixture_url (str): The URL of the fixture on FBRef.

    Returns:
        pandas.DataFrame: A DataFrame containing player stats for both teams.
    """
    try:
        # Introduce a random delay to be polite to the server
        time.sleep(random.uniform(3, 7))

        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://fbref.com/en/',
            'Connection': 'keep-alive',
        }
        
        with httpx.Client(http2=True, headers=headers, follow_redirects=True) as client:
            response = client.get(fixture_url)
            response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        all_tables = []
        
        team_tables = soup.find_all('div', class_='table_wrapper')

        all_player_stats = []

        for i, table_container in enumerate(team_tables):
            table = table_container.find('table')
            if table and 'stats_player' in table.get('id', ''):
                team_name_tag = soup.find_all('div', class_='section_heading')[i].find('h2')
                team_name = team_name_tag.get_text(strip=True).split(' Player Stats')[0]

                df = pd.read_html(str(table))[0]
                
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                
                df = df[df['Unnamed: 0_level_0_Player'].notna()]
                df = df[df['Unnamed: 0_level_0_Player'] != 'Player']
                
                df['Team'] = team_name
                all_player_stats.append(df)


        if not all_player_stats:
            print("No player stats tables found.")
            return pd.DataFrame()

        combined_df = pd.concat(all_player_stats, ignore_index=True)
        
        combined_df.columns = [re.sub(r'Unnamed: \d+_level_\d+__', '', col) for col in combined_df.columns]
        
        return combined_df

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.reason_phrase} for url: {e.request.url}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    example_url = "https://fbref.com/en/matches/a2938790/Real-Sociedad-Girona-December-12-2025-La-Liga"
    player_stats_df = scrape_fixture_player_stats(example_url)

    if not player_stats_df.empty:
        print("Successfully scraped player stats:")
        print(player_stats_df.head())
    else:
        print("Failed to scrape player stats.")