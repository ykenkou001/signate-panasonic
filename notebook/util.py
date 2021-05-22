import json


def read_json(json_path: str):
    json_load = open(json_path)
    json_file = json.load(json_load)
    return json_file
