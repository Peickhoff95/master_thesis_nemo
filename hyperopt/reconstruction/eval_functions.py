import pickle
import pandas as pd
from argparse import ArgumentParser

def unpack(x):
    if x:
        return x[0]
    return np.nan

def trials2df(trials_path: str):
    with open(trials_path, mode='rb') as file:
        trials = pickle.load(file)

    trials_df = pd.DataFrame([pd.Series(t['misc']['vals']).apply(unpack) for t in trials])
    trials_df['loss'] = [t['result']['loss'] for t in trials]
    trials_df['trial_idx'] = trials_df.index

    return trials_df

if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument('trials_file_path', type=str, help='A trials.p hyperopt file')

    args = argparser.parse_args()
    trials_df = trials2df(args.trials_file_path)
    __import__('ipdb').set_trace()
