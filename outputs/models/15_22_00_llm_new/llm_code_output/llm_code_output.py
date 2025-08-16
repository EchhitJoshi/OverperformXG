

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
