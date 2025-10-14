import os
import pickle

import pandas as pd
import ujson


def load_logfile(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as f:
        json_content = ujson.load(f)

    return pd.DataFrame(json_content)


def load_logfile_from_dir(dir: str) -> list[pd.DataFrame]:
    return [load_logfile(os.path.join(dir, filename)) for filename in os.listdir(dir) if filename.endswith(".json")]


def load_stats_from_dir(dir: str, filename: str):
    with open(os.path.join(dir, filename), "rb") as f:
        return pickle.load(f)
