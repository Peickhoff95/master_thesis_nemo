from numpy import inf
import pandas as pd
from tqdm.auto import tqdm
from evaluation import word_error_rate as cer
from argparse import ArgumentParser

if __name__ == '__main__':
   
    argparser = ArgumentParser()
    argparser.add_argument('csv_path', type=str, help='An evaluation csv file')

    args = argparser.parse_args()

    csv_path = args.csv_path
    
    df = pd.read_csv(csv_path)
    
    cer_list = []
    
    for truth, prediction in tqdm(zip(df['text'], df['predictions']), desc='Computing CER'):
        if pd.isna(prediction):
            print('NAN value encountered in denoised transcripts')
            cer_list.append(inf)
        else: 
            cer_list.append(cer.calculate_cer_from_list(truths=[truth], hypotheses=[prediction]))

    df['cer'] = cer_list

    df.to_csv(csv_path, encoding='utf-8', index=False)

    len_nan = len(df)
    df = df.dropna()
    print(f'{len_nan - len(df)} lines with NAN dropped')
    print('')

    total_cer = sum(cer_list)/len(cer_list) 

    print(f'Total CER: {total_cer}')


