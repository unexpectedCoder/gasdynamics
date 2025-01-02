import json
import os


conf = None


def load(directory: str):
    global conf
    if conf is not None:
        return
    
    path = os.path.join(directory, "settings.json")
    with open(path, "r", encoding="utf-8") as f:
        conf = json.load(f)


def get(key: str):
    return conf.get(key, None)
