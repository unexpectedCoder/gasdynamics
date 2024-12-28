import os
import yaml
from datetime import date

import config


def check(directory=os.path.dirname(__file__)):
    results = []
    check_path = os.path.join(
        directory, config.get("check_dir")
    )
    files = os.listdir(check_path)

    for fname in files:
        path = os.path.join(check_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            sol = yaml.safe_load(f)

        variant = sol["Информация"]["Вариант"]
        sol_fname = os.path.join(
            directory, config.get("solutions_dir"), f"{variant}.yml"
        )
        with open(sol_fname, "r", encoding="utf-8") as sf:
            correct_sol = yaml.safe_load(sf)
        
        res = {}
        res["Информация"] = sol["Информация"]
        del sol["Информация"]
        del correct_sol["Информация"]
        
        zipped = zip(correct_sol.items(), sol.items())
        res.update({
            key: {
                k: approx_eq(c[k], s[k]) for k in c
            }
            for ((key, c), (_, s)) in zipped
        })

        res["Проверка"] = {}
        res["Проверка"]["Дата проверки"] = date.today()
        res["Проверка"]["Результат"] = all_right(res)

        results.append(res)
    
    return results


def approx_eq(x, y):
    acc = config.get("float_eq_accuracy")
    return round(x, acc) == round(y, acc)


def all_right(checked: dict):
    d = checked.copy()
    del d["Информация"]
    del d["Проверка"]
    return all([
        v
        for sub_dict in d.values()
        for v in sub_dict.values()
    ])


def save_checked(checked: list[dict], directory: str):
    checked_dir = os.path.join(directory, config.get("checked_dir"))
    try:
        os.mkdir(checked_dir)
    except OSError:
        pass

    for c in checked:
        variant = c["Информация"]["Вариант"]
        path = os.path.join(checked_dir, f"checked_{variant}.yml")
        with open(path, "w", encoding="utf-8") as f:
            if not c["Проверка"]["Результат"]:
                c["Ошибки"] = whats_wrong(c)
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)


def whats_wrong(checked: dict):
    return [
        f"{sec}/{k}"
        for sec in checked
        for k, v in checked[sec].items()
        if not v
    ]
