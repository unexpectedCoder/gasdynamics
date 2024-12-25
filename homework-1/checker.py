import os
import yaml


DIR_NAME = os.path.dirname(__file__)
with open(os.path.join(DIR_NAME, "settings.yml"), "r") as f:
    config = yaml.safe_load(f)


def check(sol_dir: str):
    files = os.listdir(sol_dir)
    for f in files:
        # asserts
        ...


if __name__ == "__main__":
    checked = check(
        os.path.join(DIR_NAME, config["solutions_dir"])
    )
