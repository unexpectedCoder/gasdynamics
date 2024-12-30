import os
import yaml


conf = None


def load(directory: str):
    global conf
    if conf is not None:
        return
    
    path = os.path.join(directory, "settings.yml")
    with open(path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)


def get(key: str):
    return conf.get(key, None)
