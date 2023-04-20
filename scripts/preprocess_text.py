import pandas as pd
from argparse import ArgumentParser
import re

if __name__ == '__main__':

    
    argparser = ArgumentParser()
    argparser.add_argument('json', type=str, help='A config yaml file, like denoise_example.yml')

    args = argparser.parse_args()

    df = pd.read_json(args.json, lines=True)
    new_texts = []
    pattern = r'[^a-zA-Z\ ]'
    for line in df['text']:
        line = line.replace("'", " ")
        line = re.sub(pattern, '', line.strip().lower())
        line = line.replace("  ", " ")
        new_texts.append(line)
    df['text'] = new_texts
    df.to_json(args.json, orient='records', lines=True)
