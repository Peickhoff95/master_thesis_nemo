import os
import pandas as pd

def abspath(root_path, file_path):
    return os.path.join(root_path, file_path)

if __name__ == "__main__":
    files = [
        './dev.json',
        './test.json',
        './train.json',
    ]

    root_dir = '/home/ofi/data/speech/german/tuda/'
    for file in files:
        df = pd.read_json(file, lines=True) #type: pd.DataFrame
        for col in ['input', 'target']:
            df[col] = df[col].apply(lambda x: abspath(root_dir, x))

        df.to_json(file, lines=True, orient='records')
