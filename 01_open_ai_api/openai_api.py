import requests as rq
from openai import OpenAI
from gguf import GGUFReader
from pathlib import Path
import csv
from pprint import pprint

import polars as pl


QWEN_URL = 'http://127.0.0.1:8080'
GEMMA_URL = 'http://127.0.0.1:9931'

AYMAN_URL= 'http://172.26.22.147:9931'


ayman = OpenAI(
    base_url=f'{AYMAN_URL}/v1'
    , api_key='no-key'
)

r = rq.get(f'{AYMAN_URL}/health')
print(r.status_code)

# response = ayman.chat.completions.create(
#       model='local',
#       messages= [
#           {"role": "system", "content": "You are Obi-wan Kenobi, the Star Wars character."},
#           {"role": "user", "content": 'General Kenobi!'}
#           ]
# )

# print(response.choices[0].message.content)
