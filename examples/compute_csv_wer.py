from numpy import inf
import pandas as pd
from tqdm.auto import tqdm
from custom_modules import word_error_rate as wer

if __name__ == '__main__':

    csv_path = '/home/patrick/Projects/master_thesis_nemo/experiments/Conformer-Reconstruction-Unfrozen/2022-08-15_15-08-16/testset_eval.csv'
    
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

    #Eval SNR groups
    grp_snr = df.groupby(['snr'])
    dict_snr = {'denoised': [0,1]}

    for key, grp in grp_snr:
        nwer = wer.compute_wer(grp['text'], grp['noisy_prediction'])
        dwer = wer.compute_wer(grp['text'], grp['denoised_prediction'])
        dict_snr[key] = [nwer, dwer]

    df_snr = pd.DataFrame.from_dict(dict_snr)
    df_snr.to_csv(csv_path.split('.')[0]+'_snr.csv', encoding='utf-8', index=False)
    print(f'Computed WER by SNR')

    #Eval noise_condition groups
    grp_nc = df.groupby(['noise_type'])
    dict_nc = {'denoised': [0,1]}

    for key, grp in grp_nc:
        nwer = wer.compute_wer(grp['text'], grp['noisy_prediction'])
        dwer = wer.compute_wer(grp['text'], grp['denoised_prediction'])
        dict_nc[key] = [nwer, dwer]

    df_nc = pd.DataFrame.from_dict(dict_nc)
    df_nc.to_csv(csv_path.split('.')[0]+'_nc.csv', encoding='utf-8', index=False)
    print(f'Computed WER by Noise Condition')

