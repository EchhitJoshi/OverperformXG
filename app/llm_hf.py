
from data_loader import *
from transformers import pipeline
import re
import json
import pandas as pd



def ask_hf_strict(dat,question,model_id = "openai/gpt-oss-120b"):

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )


    system_prompt = f"""
    You are a careful Python soccer analyst.
    - You will receive the schema of a pandas DataFrame named `dat` (already defined at runtime).
    - Write Python code that uses ONLY `dat`, pandas (pd), and numpy (np).
    - Do NOT fabricate numbers. Compute everything from `dat`.
    - Put the final result in a variable named `result`.
    - Rows should be split by year_e field
    - Columns should be [player_id, player_name, total_games_played,
        total_minutes_played, average_rating, captain_matches,
        substitute_appearances, total_shots, shots_on_target,
        goals_scored, assists, yellow_cards, red_cards, fouls_drawn,
        fouls_committed, attempted_dribbles, successful_dribbles,
        dribbled_past, dribble_success_rate, total_passes, key_passes,
        average_pass_accuracy, total_tackles, blocks, interceptions,
        duels_contested, duels_won, duels_won_percentage, penalties_won,
        penalties_committed, penalties_scored, penalties_missed,
        penalties_saved,
        total_shots_per_90, shots_on_target_per_90,
        goals_scored_per_90, assists_per_90, yellow_cards_per_90, red_cards_per_90, fouls_drawn_per_90,
        fouls_committed_per_90, attempted_dribbles_per_90, successful_dribbles_per_90,
        dribbled_past_per_90, dribble_success_rate_per_90, total_passes_per_90, key_passes_per_90,
        average_pass_accuracy_per_90, total_tackles_per_90, blocks_per_90, interceptions_per_90,
        duels_contested_per_90, duels_won_per_90, duels_won_percentage_per_90, penalties_won_per_90,
        penalties_committed_per_90, penalties_scored_per_90, penalties_missed_per_90,
        penalties_saved_per_90,team_goals_scored, team_non_penalty_goals,
        team_goals_conceded, team_non_penalty_goals_conceded,
        matches_won]
    - Do NOT create any column names other than the list above
    - Calculate total_games_played as number of records for that year
    {'- Create _per_90 stats by weighting the respective stats by games_minutes column in data' if normalize else ''}
    - `result` must be either:
    (a) a pandas DataFrame, or
    (b) a JSON-serializable dict/list.
    - No file I/O, no network, no reading/writing disk, no imports beyond pandas/numpy.
    Return ONLY a single Python code block.
    """

    schema = {
            "columns": list(dat.columns),
            "dtypes": {c: str(dat[c].dtype) for c in dat.columns},
            "rows": len(dat)
        }

    user_prompt = f"""
    DataFrame schema:
    {json.dumps(schema, indent=2)}

    Task:
    {question}

    Requirements:
    - Use the provided `dat` (already defined in the runtime).
    - Assign your final answer to a variable named `result`.
    """

    prompt = f"{system_prompt}\nUser:{user_prompt}\nAssistant:"

    output = pipe(prompt,model = model_id,torch_dtype = 'auto')

    print(output[0])