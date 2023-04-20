from numpy import inf
import pandas as pd
from tqdm.auto import tqdm
from evaluation import word_error_rate as wer
from argparse import ArgumentParser

if __name__ == '__main__':
   
    argparser = ArgumentParser()
    argparser.add_argument('csv_path', type=str, help='An evaluation csv file')

    args = argparser.parse_args()

    csv_path = args.csv_path
    
    df = pd.read_csv(csv_path)
    
    wer_list = []
    
    for truth, prediction in tqdm(zip(df['text'], df['predictions']), desc='Computing WER'):
        if pd.isna(prediction):
            print('NAN value encountered in denoised transcripts')
            wer_list.append(wer.calculate_wer_from_list(truths=[truth], hypotheses=['']))
        else: 
            wer_list.append(wer.calculate_wer_from_list(truths=[truth], hypotheses=[prediction]))

    df['wer'] = wer_list

    df.to_csv(csv_path, encoding='utf-8', index=False)

    len_nan = len(df)
    df = df.dropna()
    print(f'{len_nan - len(df)} lines with NAN dropped')
    print('')

    total_wer = sum(wer_list)/len(wer_list) 

    print(f'Total WER: {total_wer}')


