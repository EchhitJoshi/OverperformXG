

# --- New generated code block ---
# Group data by 'year_e' and apply necessary computations
summary = dat.groupby("year_e").apply(lambda group: pd.Series({
    "player_id": group["player_id"].iloc[0],
    "player_name": group["player_name"].iloc[0],
    "total_games_played": group["games_number"].count(),
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
    "average_pass_accuracy_per_90": group["pass_accuracy_perc"].mean() * 90,
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

# Reset index to clean up result DataFrame
summary.reset_index(inplace=True)

# Assign result
result = summary
# --- End of generated code block ---


# --- New generated code block ---
# Import necessary libraries
import pandas as pd
import numpy as np

# Grouping data by player_id, player_name, and year_e, then calculating required aggregates
player_summary = (
    dat.groupby(['year_e', 'player_id', 'player_name'], as_index=False)
    .agg(
        total_games_played=('games_number', 'count'),
        total_minutes_played=('games_minutes', 'sum'),
        average_rating=('games_rating', 'mean'),
        captain_matches=('games_captain', 'sum'),
        substitute_appearances=('games_substitute', 'sum'),
        total_shots=('shots_total', 'sum'),
        shots_on_target=('shots_on', 'sum'),
        goals_scored=('goals_total', 'sum'),
        assists=('goals_assists', 'sum'),
        yellow_cards=('cards_yellow', 'sum'),
        red_cards=('cards_red', 'sum'),
        fouls_drawn=('fouls_drawn', 'sum'),
        fouls_committed=('fouls_committed', 'sum'),
        attempted_dribbles=('dribbles_attempts', 'sum'),
        successful_dribbles=('dribbles_success', 'sum'),
        dribbled_past=('dribbles_past', 'sum'),
        dribble_success_rate=('dribble_success_rate', 'mean'),  # Average rate
        total_passes=('passes_total', 'sum'),
        key_passes=('passes_key', 'sum'),
        average_passes_accurate=('passes_accurate', 'mean'),
        average_pass_accuracy=('pass_accuracy_perc', 'mean'),
        total_tackles=('tackles_total', 'sum'),
        blocks=('tackles_blocks', 'sum'),
        interceptions=('tackles_interceptions', 'sum'),
        duels_contested=('duels_total', 'sum'),
        duels_won=('duels_won', 'sum'),
        duels_won_percentage=('duels_won_perc', 'mean'),  # Average percentage
        penalties_won=('penalty_won', 'sum'),
        penalties_committed=('penalty_commited', 'sum'),
        penalties_scored=('penalty_scored', 'sum'),
        penalties_missed=('penalty_missed', 'sum'),
        penalties_saved=('penalty_saved', 'sum'),
        team_goals_scored=('team_goals_scored', 'sum'),
        team_non_penalty_goals=('team_non_penalty_goals_scored', 'sum'),
        team_goals_conceded=('team_goals_conceded', 'sum'),
        team_non_penalty_goals_conceded=('team_non_penalty_goals_conceded', 'sum'),
        matches_won=('win', 'sum'),
    )
)

# Calculations for per 90-minute stats
per_90_fields = [
    'total_shots', 'shots_on_target', 'goals_scored', 'assists',
    'yellow_cards', 'red_cards', 'fouls_drawn', 'fouls_committed',
    'attempted_dribbles', 'successful_dribbles', 'dribbled_past',
    'total_passes', 'key_passes', 'average_passes_accurate',
    'total_tackles', 'blocks', 'interceptions', 'duels_contested',
    'duels_won', 'penalties_won', 'penalties_committed',
    'penalties_scored', 'penalties_missed', 'penalties_saved'
]

# Adding per 90-minute stats
for field in per_90_fields:
    player_summary[f'{field}_per_90'] = (
        player_summary[field] / player_summary['total_minutes_played'] * 90
    )

# Per 90-minute adjustments for percentages
player_summary['duels_won_percentage_per_90'] = player_summary['duels_won_percentage']
player_summary['dribble_success_rate_per_90'] = player_summary['dribble_success_rate']
player_summary['average_pass_accuracy_per_90'] = player_summary['average_pass_accuracy']

# Sorting values
player_summary = player_summary.sort_values(by=['year_e', 'player_id']).reset_index(drop=True)

# Final result in the specified format
result = player_summary
# --- End of generated code block ---



# --- New generated code block ---
import plotly.express as px

fig = px.histogram(
    dat,
    x='games_rating',
    facet_col='year_e',
    title='Histogram of Game Ratings Faceted by Year',
    labels={'games_rating': 'Game Ratings', 'year_e': 'Year'},
    nbins=20,  # Adjust the number of bins as needed
    color_discrete_sequence=['blue']  # You can adjust the color if desired
)

# Update layout for better visuals
fig.update_layout(
    bargap=0.2,  # Adjust the gap between bars
    xaxis_title='Game Rating',
    yaxis_title='Count',
    showlegend=False
)

fig.show()
# --- End of generated code block ---



# --- New generated code block ---
import plotly.express as px
import pandas as pd

# Convert 'month_e' to a string for compatibility with Plotly
dat['month_e_str'] = dat['month_e'].astype(str)

# Group by 'month_e_str' and calculate the average of 'games_rating'
average_rating_by_month = dat.groupby('month_e_str', as_index=False)['games_rating'].mean()

# Create a line plot
fig = px.line(average_rating_by_month, 
              x='month_e_str', 
              y='games_rating', 
              title='Average Games Rating by Month',
              labels={'month_e_str': 'Month', 'games_rating': 'Average Games Rating'})

fig.show()
# --- End of generated code block ---


# --- New generated code block ---
import plotly.express as px
import pandas as pd

# Ensure month_e (Period dtype) is converted to a format compatible with Plotly
dat['month_e'] = dat['month_e'].astype(str)  # Convert Period[M] to string

# Compute the average games_rating by month_e and team
average_rating = dat.groupby(['team', 'month_e'])['games_rating'].mean().reset_index()

# Generate the plot
fig = px.line(
    average_rating,
    x='month_e',
    y='games_rating',
    color='team',
    facet_col='team',
    title='Average Games Rating by Month Faceted by Team',
    labels={"games_rating": "Average Rating", "month_e": "Month"}
)

fig.show()
# --- End of generated code block ---


# --- New generated code block ---
import pandas as pd
import plotly.express as px

# Convert 'month_e' (Period) to string to make it compatible with Plotly
dat['month_e_str'] = dat['month_e'].astype(str)

# Compute average games_rating and standard deviation grouped by month_e_str and team
agg_data = dat.groupby(['month_e_str', 'team']).agg(
    avg_games_rating=('games_rating', 'mean'),
    std_games_rating=('games_rating', 'std')
).reset_index()

# Create the scatter plot with error cloud faceted by team
fig = px.scatter(
    agg_data,
    x='month_e_str',
    y='avg_games_rating',
    error_y='std_games_rating',
    facet_col='team',
    labels={'month_e_str': 'Month', 'avg_games_rating': 'Average Games Rating'},
    title='Average Games Rating by Month (faceted by Team)',
    height=600
)

# Update layout for better aesthetics
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Average Games Rating',
    title_x=0.5
)

# Display the plot
fig.show()
# --- End of generated code block ---






# --- New generated code block ---
# Grouping data by fixture_id and calculating relevant statistics
result = dat.groupby('fixture_id').agg(
    total_offsides=('offsides', 'sum'),
    total_games_minutes=('games_minutes', 'sum'),
    avg_games_minutes=('games_minutes', 'mean'),
    total_shots=('shots_total', 'sum'),
    avg_shots=('shots_total', 'mean'),
    total_goals=('goals_total', 'sum'),
    total_assists=('goals_assists', 'sum'),
    total_saves=('goals_saves', 'sum'),
    total_passes=('passes_total', 'sum'),
    key_passes=('passes_key', 'sum'),
    accurate_passes=('passes_accurate', 'sum'),
    total_tackles=('tackles_total', 'sum'),
    total_blocks=('tackles_blocks', 'sum'),
    interceptions=('tackles_interceptions', 'sum'),
    total_duels=('duels_total', 'sum'),
    duels_won=('duels_won', 'sum'),
    total_dribbles=('dribbles_attempts', 'sum'),
    successful_dribbles=('dribbles_success', 'sum'),
    fouls_drawn=('fouls_drawn', 'sum'),
    fouls_committed=('fouls_committed', 'sum'),
    yellow_cards=('cards_yellow', 'sum'),
    red_cards=('cards_red', 'sum'),
    penalties_won=('penalty_won', 'sum'),
    penalties_committed=('penalty_commited', 'sum'),
    penalties_scored=('penalty_scored', 'sum'),
    penalties_missed=('penalty_missed', 'sum'),
    penalties_saved=('penalty_saved', 'sum'),
    total_team_goals=('team_goals_scored', 'sum'),
    avg_team_goals=('team_goals_scored', 'mean'),
    total_team_goals_conceded=('team_goals_conceded', 'sum'),
    avg_team_goals_conceded=('team_goals_conceded', 'mean'),
    total_non_penalty_team_goals=('team_non_penalty_goals_scored', 'sum'),
    avg_non_penalty_team_goals=('team_non_penalty_goals_scored', 'mean'),
    avg_game_rating=('games_rating', 'mean'),
    total_captains=('games_captain', 'sum'),
    total_substitutes=('games_substitute', 'sum'),
    fixture_date=('fixture_date', 'first'),  # Keeping the corresponding date for each fixture
)

# Resetting the index to make fixture_id a column
result = result.reset_index()
# --- End of generated code block ---


# --- New generated code block ---
# Grouping by the fixture_id to calculate statistics for each fixture
per_fixture_stats = dat.groupby("fixture_id").agg(
    player_count=("player_id", "nunique"),
    avg_game_minutes=("games_minutes", "mean"),
    avg_game_rating=("games_rating", "mean"),
    total_offsides=("offsides", "sum"),
    total_shots=("shots_total", "sum"),
    total_shots_on_target=("shots_on", "sum"),
    total_goals=("goals_total", "sum"),
    total_assists=("goals_assists", "sum"),
    total_saves=("goals_saves", "sum"),
    total_passes=("passes_total", "sum"),
    total_key_passes=("passes_key", "sum"),
    total_accurate_passes=("passes_accurate", "sum"),
    total_tackles=("tackles_total", "sum"),
    total_blocks=("tackles_blocks", "sum"),
    total_interceptions=("tackles_interceptions", "sum"),
    total_duels=("duels_total", "sum"),
    total_duels_won=("duels_won", "sum"),
    total_dribbles_attempts=("dribbles_attempts", "sum"),
    total_dribbles_success=("dribbles_success", "sum"),
    total_dribbles_past=("dribbles_past", "sum"),
    total_fouls_drawn=("fouls_drawn", "sum"),
    total_fouls_committed=("fouls_committed", "sum"),
    yellow_cards=("cards_yellow", "sum"),
    red_cards=("cards_red", "sum"),
    total_penalty_won=("penalty_won", "sum"),
    total_penalty_commited=("penalty_commited", "sum"),
    total_penalty_scored=("penalty_scored", "sum"),
    total_penalty_missed=("penalty_missed", "sum"),
    avg_dribble_success_rate=("dribble_success_rate", "mean"),
    avg_target_shot_conversion_perc=("target_shot_conversion_perc", "mean"),
    avg_duels_won_perc=("duels_won_perc", "mean"),
    avg_pass_accuracy_perc=("pass_accuracy_perc", "mean")
).reset_index()

# Selecting 'team_' columns which are already at the fixture level
team_columns = dat[[
    "fixture_id", 
    "team_goals_scored", 
    "team_non_penalty_goals_scored", 
    "team_goals_scored_half",
    "team_goals_conceded",
    "team_non_penalty_goals_conceded",
    "team_goals_conceded_half",
    "opponent",
    "fixture_date",
    "team_winner",
    "team",
    "outcome"
]].drop_duplicates(subset="fixture_id")

# Merging the per-fixture stats with team-level data
result = pd.merge(per_fixture_stats, team_columns, on="fixture_id", how="left")
# --- End of generated code block ---


# --- New generated code block ---
# Group by 'fixture_id' and 'team' and calculate relevant aggregate statistics.
# Columns starting with 'team_' are directly included in the final DataFrame without further aggregation.

# Columns to calculate aggregate statistics (sum and mean where relevant)
aggregate_columns = {
    "player_id": "count",  # Count of players
    "offsides": "sum",
    "games_minutes": "sum",
    "games_rating": "mean",
    "shots_total": "sum",
    "shots_on": "sum",
    "goals_total": "sum",
    "goals_assists": "sum",
    "goals_saves": "sum",
    "passes_total": "sum",
    "passes_key": "sum",
    "passes_accurate": "sum",
    "tackles_total": "sum",
    "tackles_blocks": "sum",
    "tackles_interceptions": "sum",
    "duels_total": "sum",
    "duels_won": "sum",
    "dribbles_attempts": "sum",
    "dribbles_success": "sum",
    "dribbles_past": "sum",
    "fouls_drawn": "sum",
    "fouls_committed": "sum",
    "cards_yellow": "sum",
    "cards_red": "sum",
    "penalty_won": "sum",
    "penalty_commited": "sum",
    "penalty_scored": "sum",
    "penalty_missed": "sum",
    "penalty_saved": "sum",
    "dribble_success_rate": "mean",
    "target_shot_conversion_perc": "mean",
    "duels_won_perc": "mean",
    "pass_accuracy_perc": "mean",
    "win": "unique"
}

# Grouping the data by fixture_id and team, and aggregating the relevant statistics.
grouped_data = dat.groupby(['fixture_id', 'team']).agg(aggregate_columns).reset_index()

# Selecting 'team_' columns directly from the original data to avoid aggregation.
team_columns = [
    'team_goals_scored',
    'team_non_penalty_goals_scored',
    'team_goals_scored_half',
    'team_goals_conceded',
    'team_non_penalty_goals_conceded',
    'team_goals_conceded_half',
    'team_winner'
]
first_team_columns = dat.groupby(['fixture_id', 'team'])[team_columns].first().reset_index()

# Merging aggregated data with the 'team_' columns.
result = pd.merge(grouped_data, first_team_columns, on=['fixture_id', 'team'])
# --- End of generated code block ---


# --- New generated code block ---
# Grouping relevant statistics by FIXTURE_ID and TEAM, and calculating stats for each games_position as well.
# Exclude 'team_' columns from aggregation as instructed.

# Columns to be aggregated
aggregate_columns = [
    "offsides", "games_minutes", "games_number", "games_rating", "shots_total", "shots_on",
    "goals_total", "goals_assists", "goals_saves", "passes_total", "passes_key", "passes_accurate",
    "tackles_total", "tackles_blocks", "tackles_interceptions", "duels_total", "duels_won",
    "dribbles_attempts", "dribbles_success", "dribbles_past", "fouls_drawn", "fouls_committed",
    "cards_yellow", "cards_red", "penalty_won", "penalty_commited", "penalty_scored",
    "penalty_missed", "penalty_saved", "dribble_success_rate", "target_shot_conversion_perc",
    "duels_won_perc", "pass_accuracy_perc"
]

# Aggregation functions to be applied
aggregation_functions = {col: ["sum", "mean"] for col in aggregate_columns}

# Grouping by FIXTURE_ID and TEAM for aggregate stats
stat_per_fixture_team = dat.groupby(["fixture_id", "team"]).agg(aggregation_functions)

# Flattening MultiIndex columns
stat_per_fixture_team.columns = ['_'.join(col).rstrip('_') for col in stat_per_fixture_team.columns]

# Reset index to get a flat structure
stat_per_fixture_team = stat_per_fixture_team.reset_index()

# Aggregating stats per games_position
stat_per_position = dat.groupby("games_position").agg(aggregation_functions)

# Flattening MultiIndex columns
stat_per_position.columns = ['_'.join(col).rstrip('_') for col in stat_per_position.columns]

# Renaming columns specifically for games_position to differentiate them
stat_per_position = stat_per_position.reset_index()
stat_per_position.columns = ['games_position' if col == 'games_position' else f"{col}_by_position" for col in stat_per_position.columns]

# Merging the statistics from fixture+team and position into a single output dictionary for clarity
result = {
    "per_fixture_team": stat_per_fixture_team,
    "per_games_position": stat_per_position
}
# --- End of generated code block ---



# --- New generated code block ---
import pandas as pd
import numpy as np

# Helper function to dynamically create aggregated column names based on positions
def position_column_aggregator(stat, position):
    return f"{stat}_{position}"

# Create a dictionary of aggregation functions for player stats 
stats_to_aggregate = [
    "games_minutes", "games_number", "games_rating", "shots_total", "shots_on",
    "goals_total", "goals_assists", "goals_saves", "passes_total", "passes_key",
    "passes_accurate", "tackles_total", "tackles_blocks", "tackles_interceptions",
    "duels_total", "duels_won", "dribbles_attempts", "dribbles_success", "dribbles_past",
    "fouls_drawn", "fouls_committed", "cards_yellow", "cards_red", "penalty_won",
    "penalty_commited", "penalty_scored", "penalty_missed", "penalty_saved"
]

# Create aggregations for overall and position-based statistics
aggregations = {}
positions = dat["games_position"].dropna().unique()  # Get unique positions

for stat in stats_to_aggregate:
    aggregations[stat] = ['sum', 'mean']  # Overall statistical aggregation
    for position in positions:
        # Create conditional aggregation for each position subset
        aggregations[position_column_aggregator(stat, position)] = (
            lambda x, stat=stat, pos=position: x[dat.loc[x.index, "games_position"] == pos].sum()
        )

# Aggregate the DataFrame grouped by fixture_id and team
grouped = dat.groupby(["fixture_id", "team"]).agg(aggregations)

# Flatten multi-level column names to readable format
grouped.columns = [
    f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != "" else col[0]
    for col in grouped.columns
]

# Add team-level columns (unchanged aggregation)
team_columns = [
    "team_goals_scored", "team_non_penalty_goals_scored", "team_goals_scored_half",
    "team_goals_conceded", "team_non_penalty_goals_conceded", "team_goals_conceded_half",
    "team_winner", "opponent", "fixture_date"
]
result = grouped.reset_index()[["fixture_id", "team"] + team_columns + list(grouped.columns)]
# --- End of generated code block ---


# --- New generated code block ---
# Filter columns indicating team-specific stats; they'll remain unaggregated.
team_columns = [col for col in dat.columns if col.startswith('team_')]

# Generate aggregations per fixture/team/games_position.
agg_results = (
    dat
    .groupby(['fixture_id', 'team', 'games_position'])
    .agg({col: ['sum', 'mean'] for col in dat.columns if col not in ['fixture_id', 'team', 'games_position'] + team_columns})
    .reset_index()
)

# Flatten multi-index columns into a more readable format `statistic_games_position_value`.
agg_results.columns = [
    f'{col[0]}_{col[1]}_{col[2]}' if col[1] else col[0]
    for col in agg_results.columns.to_flat_index()
]

# Include unaggregated team-level columns per fixture/team, excluding games_position.
team_level_data = dat[['fixture_id', 'team'] + team_columns].drop_duplicates()

# Merge aggregated stats with team-level data.
result_df = pd.merge(
    agg_results,
    team_level_data,
    on=['fixture_id', 'team'],
    how='left'
)

result = result_df
# --- End of generated code block ---


# --- New generated code block ---
# Group the data by 'fixture_id', 'team', and 'games_position'
aggregations = {
    'offsides': ['sum', 'mean'],
    'games_minutes': ['sum', 'mean'],
    'games_number': ['sum', 'mean'],
    'games_rating': ['mean'],
    'games_captain': ['sum', 'mean'],
    'games_substitute': ['sum', 'mean'],
    'shots_total': ['sum', 'mean'],
    'shots_on': ['sum', 'mean'],
    'goals_total': ['sum', 'mean'],
    'goals_assists': ['sum', 'mean'],
    'goals_saves': ['sum', 'mean'],
    'passes_total': ['sum', 'mean'],
    'passes_key': ['sum', 'mean'],
    'passes_accurate': ['sum', 'mean'],
    'tackles_total': ['sum', 'mean'],
    'tackles_blocks': ['sum', 'mean'],
    'tackles_interceptions': ['sum', 'mean'],
    'duels_total': ['sum', 'mean'],
    'duels_won': ['sum', 'mean'],
    'dribbles_attempts': ['sum', 'mean'],
    'dribbles_success': ['sum', 'mean'],
    'dribbles_past': ['sum', 'mean'],
    'fouls_drawn': ['sum', 'mean'],
    'fouls_committed': ['sum', 'mean'],
    'cards_yellow': ['sum', 'mean'],
    'cards_red': ['sum', 'mean'],
    'penalty_won': ['sum', 'mean'],
    'penalty_commited': ['sum', 'mean'],
    'penalty_scored': ['sum', 'mean'],
    'penalty_missed': ['sum', 'mean'],
    'penalty_saved': ['sum', 'mean'],
    'dribble_success_rate': ['mean'],
    'target_shot_conversion_perc': ['mean'],
    'duels_won_perc': ['mean'],
    'pass_accuracy_perc': ['mean']
}

# Perform the groupby and aggregation
aggregated = dat[dat.games_position.isin("D","M","F")].groupby(['fixture_id', 'team', 'games_position']).agg(aggregations)

# Flatten column names to remove multi-index for aggregated statistics
aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]

# Reset index for pivoting/grouped analysis
aggregated = aggregated.reset_index()

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
result = merged_data.pivot_table(
    index=['fixture_id', 'team'],  # Group by fixture_id and team
    columns='games_position',     # Separate columns by games_position
    values=[col for col in merged_data.columns if col not in ['fixture_id', 'team', 'games_position']],
    aggfunc='first'               # Only one value per attribute for each position
)

# Flatten the resulting pivot table's columns
result.columns = ['_'.join([str(c) for c in col]).strip() for col in result.columns]

# Reset index to have fixture_id and team as columns
result = result.reset_index()
# --- End of generated code block ---
