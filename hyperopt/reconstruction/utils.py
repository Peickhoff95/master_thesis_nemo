from typing import List
import pandas as pd

__all__=['load_nemo_json', 'load_nemo_jsons']

def load_nemo_json(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_json(path, orient='records', lines=True)
    return df

def load_nemo_jsons(paths: List[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for path in paths:
        dfs.append(load_nemo_json(path))

    return pd.concat(dfs)
