from numpy import inf
import pandas as pd
from tqdm.auto import tqdm
from custom_modules import word_error_rate as wer

if __name__ == '__main__':

    csv_path = '/home/patrick/Projects/master_thesis_nemo/experiments/2022-07-09_03-09-18/trainset_28spk_eval.csv'
    
    df = pd.read_csv(csv_path)
    
    wer_denoised = []
    wer_noisy = []
    wer_diff = []
    
    for truth, denoised_prediction, noisy_prediction in tqdm(zip(df['text'], df['denoised_prediction'], df['noisy_prediction']), desc='Computing WER'):
        if pd.isna(denoised_prediction):
            print('NAN value encountered in denoised transcripts')
            wer_denoised.append(inf)
        else:
            wer_denoised.append(wer.compute_wer(truths=[truth], predictions=[denoised_prediction]))
        if pd.isna(noisy_prediction):
            print('NAN value encountered in noisy transcripts')
            wer_noisy.append(inf)
        else:
            wer_noisy.append(wer.compute_wer(truths=[truth], predictions=[noisy_prediction]))
        wer_diff.append(wer_denoised[-1] - wer_noisy[-1])

    df['wer_denoised'] = wer_denoised
    df['wer_noisy'] = wer_noisy
    df['wer_diff'] = wer_diff

    df.to_csv(csv_path, encoding='utf-8', index=False)

    len_nan = len(df)
    df = df.dropna()
    print(f'{len_nan - len(df)} lines with NAN dropped')
    print('')

    total_wer_denoised = wer.compute_wer(truths=df['text'], predictions=df['denoised_prediction'])
    total_wer_noisy = wer.compute_wer(truths=df['text'], predictions=df['noisy_prediction'])
    total_wer_diff = total_wer_denoised - total_wer_noisy

    print(f'Total WER denoised: {total_wer_denoised}')
    print(f'Total WER noisy: {total_wer_noisy}')
    print(f'Total WER diff: {total_wer_diff}')
