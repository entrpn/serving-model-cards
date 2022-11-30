from os import path
import json

def readTextFile(*args):
    dir = path.dirname(__file__)
    entry = path.join(dir, *args)
    with open(entry, "r", encoding="utf8") as f:
        data = f.read()
    return data

def js():
    data = readTextFile("js", "index.js")
    data = "(z) => {" + data + "; return z ?? [] }"
    return data

def call_js(sd_method, **kwargs):
    param_str = json.dumps(kwargs)
    return f"async (x) => {{ return await SD.{sd_method}({{ x, ...{param_str} }}) ?? []; }}"