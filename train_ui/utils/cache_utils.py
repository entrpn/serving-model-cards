import json
from os.path import exists
import os

def save_ui_values(cache_file, **kwargs):
    current_values = {}
    os.makedirs(os.sep.join(cache_file.split(os.sep)[:-1]),exist_ok=True)
    if exists(cache_file):
        with open(cache_file,'r') as f:
            current_values = json.loads(f.read())
    current_values.update(kwargs)
    with open(cache_file,'w') as f:
        f.write(json.dumps(current_values))
