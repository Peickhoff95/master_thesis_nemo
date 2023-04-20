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
    
    wer_denoised = []
    wer_noisy = []
    wer_diff = []
    
    for truth, denoised_prediction, noisy_prediction in tqdm(zip(df['text'], df['denoised_prediction'], df['noisy_prediction']), desc='Computing WER'):
        if pd.isna(denoised_prediction):
            print('NAN value encountered in denoised transcripts')
            wer_denoised.append(inf)
        else:
            wer_denoised.append(wer.calculate_wer_from_list(truths=[truth], hypotheses=[denoised_prediction]))
        if pd.isna(noisy_prediction):
            print('NAN value encountered in noisy transcripts')
            wer_noisy.append(inf)
        else:
            wer_noisy.append(wer.calculate_wer_from_list(truths=[truth], hypotheses=[noisy_prediction]))
        wer_diff.append(wer_denoised[-1] - wer_noisy[-1])

    df['wer_denoised'] = wer_denoised
    df['wer_noisy'] = wer_noisy
    df['wer_diff'] = wer_diff

    df.to_csv(csv_path, encoding='utf-8', index=False)

    len_nan = len(df)
    df = df.dropna()
    print(f'{len_nan - len(df)} lines with NAN dropped')
    print('')

   # total_wer_denoised = wer.calculate_wer_from_list(truths=df['text'], hypotheses=df['denoised_prediction'])
   # total_wer_noisy = wer.calculate_wer_from_list(truths=df['text'], hypotheses=df['noisy_prediction'])
   # total_wer_diff = total_wer_denoised - total_wer_noisy

   # print(f'Total WER denoised: {total_wer_denoised}')
   # print(f'Total WER noisy: {total_wer_noisy}')
   # print(f'Total WER diff: {total_wer_diff}')

    print(f'Total WER denoised: {df["wer_denoised"].mean()}')
    print(f'Total WER noisy: {df["wer_noisy"].mean()}')
    print(f'Total WER diff: {df["wer_denoised"].mean() - df["wer_noisy"].mean()}')
   
   #Eval SNR groups
   # grp_snr = df.groupby(['snr'])
   # dict_snr = {'denoised': [0,1]}

   # for key, grp in grp_snr:
   #     nwer = wer.calculate_wer_from_list(grp['text'], grp['noisy_prediction'])
   #     dwer = wer.calculate_wer_from_list(grp['text'], grp['denoised_prediction'])
   #     dict_snr[key] = [nwer, dwer]

   # df_snr = pd.DataFrame.from_dict(dict_snr)
   # df_snr.to_csv(csv_path.split('.')[0]+'_snr.csv', encoding='utf-8', index=False)
   # print(f'Computed WER by SNR')

   # #Eval noise_condition groups
   # grp_nc = df.groupby(['noise_type'])
   # dict_nc = {'denoised': [0,1]}

   # for key, grp in grp_nc:
   #     nwer = wer.calculate_wer_from_list(grp['text'], grp['noisy_prediction'])
   #     dwer = wer.calculate_wer_from_list(grp['text'], grp['denoised_prediction'])
   #     dict_nc[key] = [nwer, dwer]

   # df_nc = pd.DataFrame.from_dict(dict_nc)
   # df_nc.to_csv(csv_path.split('.')[0]+'_nc.csv', encoding='utf-8', index=False)
   # print(f'Computed WER by Noise Condition')

