
from data_loader import *
from openai import OpenAI
import re
import json

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
    
