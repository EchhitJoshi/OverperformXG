
from data_loader import *
import folder_manager
from openai import OpenAI
from transformers import pipeline
import re
import json
import pandas as pd


with open("config.yaml",'r') as file:
    config = yaml.safe_load(file)

token = config['OPENAI_TOKEN']
open_ai_key = config['OPENAI_KEY']
endpoint = "https://models.github.ai/inference"
model = ["gpt-4o","gpt-4o-mini","gpt-3.5-turbo"]


def ask_gpt(dat, question):

    """
    dat: pandas DataFrame to ask gpt a question about
    question: question to ask of the data
    """

    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant who always responds with Python code.
                The response must be valid JSON object with Key value pairs."""
            },
            {
                "role": "user",
                "content": f"""
                
                Given this data: {dat.to_csv(index = False)}

                
                please answer:
                {question}
                
                """,
            }
        ],
        response_format = {"type":"json_object"},
        model=model
    )

    
    data = json.loads(response.choices[0].message.content)
    print("Raw GPT JSON:", data)

    # ---- Normalization ----
    if isinstance(data, dict):
        # If all values are scalars → wrap in list
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            data = [data]  # becomes list of dicts
    elif isinstance(data, list):
        # If GPT returned list of scalars (unlikely), wrap in dict
        if all(not isinstance(v, (list, dict)) for v in data):
            data = [{"value": v} for v in data]

    df = pd.DataFrame(data)
    return df
    



def ask_gpt_q(question:str,model = 'gpt-4o'):

    """
    question: str question to ask an LLM model
    """

    client = OpenAI(
        base_url = endpoint,
        api_key = token
    )


    response = client.chat.completions.create(
        messages = [
            {
                "role":"system",
                "content":"""You are a basic chatbot that answers with a how sure you are of your answer as a probability in the json form {answer: answer, certainty: probabilty}.The response must be valid JSON object with Key value pairs."""
            },
            {
                "role":"user",
                "content":f"""{question}"""
            }

        ],
        response_format = {'type':'json_object'},
        model = model
    )

    data = json.loads(response.choices[0].message.content)

    if isinstance(data,dict):
        if all(not isinstance(v,(list,dict)) for v in data.values()):
            data = [data]
    elif isinstance(data,list):
        if all(not isinstance(v,(list,dict)) for v in data):
            data = [{'value':v} for v in data]

    df = pd.DataFrame(data)
    return df

        

def ask_gpt_strict(dat: pd.DataFrame, question: str,normalize = False):
    client = OpenAI(base_url=endpoint, api_key=token)
    
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
       average_passes_accurate, average_pass_accuracy, total_tackles, blocks, interceptions,
       duels_contested, duels_won,duels_won_percentage, penalties_won,
       penalties_committed, penalties_scored, penalties_missed,
       penalties_saved,
       total_shots_per_90, shots_on_target_per_90,
       goals_scored_per_90, assists_per_90, yellow_cards_per_90, red_cards_per_90, fouls_drawn_per_90,
       fouls_committed_per_90, attempted_dribbles_per_90, successful_dribbles_per_90,
       dribbled_past_per_90, dribble_success_rate_per_90, total_passes_per_90, key_passes_per_90,
       average_passes_accurate_per_90, average_pass_accuracy_per_90,total_tackles_per_90, blocks_per_90, interceptions_per_90,
       duels_contested_per_90, duels_won_per_90, duels_won_percentage_per_90, penalties_won_per_90,
       penalties_committed_per_90, penalties_scored_per_90, penalties_missed_per_90,
       penalties_saved_per_90,team_goals_scored, team_non_penalty_goals,
       team_goals_conceded, team_non_penalty_goals_conceded,
       matches_won]
- Do NOT create any column names other than the list above
- Calculate total_games_played as number of records for that year
- _perc are the only percentage columns so match that with percentage column calculations
{'- Create _per_90 stats by weighting the respective stats by games_minutes column in data' if normalize else ''}
- `result` must be either:
  (a) a pandas DataFrame, or
  (b) a JSON-serializable dict/list.
- No file I/O, no network, no reading/writing disk, no imports beyond pandas/numpy.
Return ONLY a single Python code block.
"""
#{'- Create _per_90 stats after computing non _per_90 columns and then by dividing the stats by (total_minutes_played/90)' if normalize else ''}

    schema = {
        "columns": list(dat.columns),
        "dtypes": {c: str(dat[c].dtype) for c in dat.columns},
        "rows": len(dat)
    }


    #print(schema)

    user_prompt = f"""
DataFrame schema:
{json.dumps(schema, indent=2)}

Task:
{question}

Requirements:
- Use the provided `dat` (already defined in the runtime).
- Assign your final answer to a variable named `result`.
"""

    # One call, no JSON mode (we want code)
    try:
        print(f"trying {model[0]}")
        resp = client.chat.completions.create(
            model=model[0],  # e.g., "gpt-4o" or "gpt-4o-mini"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        try:
            print(f"trying {model[1]}")
            resp = client.chat.completions.create(
                model=model[1],  # e.g., "gpt-4o" or "gpt-4o-mini"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as e:
            print(f"trying {model[2]}")
            resp = client.chat.completions.create(
            model=model[2],  # e.g., "gpt-4o" or "gpt-4o-mini"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
            )

    
    content = resp.choices[0].message.content

    # Extract code from a ```python ... ``` block
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", content)
    code = m.group(1).strip() if m else content.strip()

    with open(f"{folder_manager.llm_code_path}llm_code_output.py", "a", encoding="utf-8") as f:
        f.write("\n\n# --- New generated code block ---\n")
        f.write(code)
        f.write("\n# --- End of generated code block ---\n")


    # Naive safety checks (keep it simple but useful)
    forbidden = ("open(", "os.", "sys.", "subprocess", "requests", "eval(", "exec(", "__import__")
    if any(x in code for x in forbidden):
        raise ValueError(f"Model attempted {code} unsafe operations. Aborting.")

    # Execute the code with a restricted namespace
    local_vars = {"pd": pd, "np": np, "dat": dat}
    exec(code, local_vars, local_vars)

    if "result" not in local_vars:
        raise ValueError("No `result` variable produced by model.")

    result = local_vars["result"]

    # Normalize to DataFrame if possible
    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, (list, dict)):
        try:
            return pd.DataFrame(result)
        except Exception:
            # Fallback: return as-is if it doesn't tabulate cleanly
            return result
    else:
        # Numbers/strings, etc.
        return result





def compare_players_from_llm(dat:pd.DataFrame,player_list:list,years:list,normalize= False):

    players_dat = {
        player: dat[(dat.player_name == player) & (dat.year_e.isin(years))] for player in player_list
    }

    players_summary = dict()
    for player in player_list:
        print(f"Checking for {player}")
        players_summary[player] = ask_gpt_strict(players_dat[player],"give me a comprehensive summary for this player",normalize)
        time.sleep(60)
    
    
    # players_summary = {
    #     player: ask_gpt_strict(players_dat[player],"give me a comprehensive summary for this player") for player in player_list
    # }

    final_dat = pd.DataFrame()
    for player_data in players_summary.values():
        final_dat = pd.concat([final_dat,player_data],axis = 0)
    
    final_dat = final_dat.T
    final_dat.columns = final_dat.loc['player_name'].values
    final_dat.drop(index = 'player_name',inplace = True)
    final_dat.index.name = 'stat_type'

    return final_dat
















