
from data_loader import *
from openai import OpenAI
import re
import json
import pandas as pd

with open("config.yaml",'r') as file:
    config = yaml.safe_load(file)

token = config['OPENAI_TOKEN']
endpoint = "https://models.github.ai/inference"
model = "gpt-4o"


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

        

def ask_gpt_strict(dat: pd.DataFrame, question: str):
    client = OpenAI(base_url=endpoint, api_key=token)

    system_prompt = """
You are a careful Python soccer analyst.
- You will receive the schema of a pandas DataFrame named `df` (already defined at runtime).
- Write Python code that uses ONLY `df`, pandas (pd), and numpy (np).
- Do NOT fabricate numbers. Compute everything from `df`.
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
       penalties_saved, team_goals_scored, team_non_penalty_goals,
       team_goals_conceded, team_non_penalty_goals_conceded,
       matches_won]
- Do NOT Change any column names
- Calculate total_games_played as number of records for that year
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
- Use the provided `df` (already defined in the runtime).
- Assign your final answer to a variable named `result`.
"""

    # One call, no JSON mode (we want code)
    resp = client.chat.completions.create(
        model=model,  # e.g., "gpt-4o" or "gpt-4o-mini"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = resp.choices[0].message.content

    # Extract code from a ```python ... ``` block
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", content)
    code = m.group(1).strip() if m else content.strip()

    # Naive safety checks (keep it simple but useful)
    forbidden = ("import ", "open(", "os.", "sys.", "subprocess", "requests", "eval(", "exec(", "__import__")
    if any(x in code for x in forbidden):
        raise ValueError("Model attempted unsafe operations. Aborting.")

    # Execute the code with a restricted namespace
    local_vars = {"pd": pd, "np": np, "df": dat}
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


















