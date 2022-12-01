import json
from os.path import exists

def save_ui_values(cache_file, **kwargs):
    current_values = {}
    if exists(cache_file):
        with open(cache_file,'r') as f:
            current_values = json.loads(f.read())
    current_values.update(kwargs)
    with open(cache_file,'w') as f:
        f.write(json.dumps(current_values))
