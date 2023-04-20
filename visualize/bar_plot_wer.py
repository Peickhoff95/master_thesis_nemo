import pandas as pd
from argparse import ArgumentParser
from matplotlib import pyplot as plt

if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument('noisy', type=str, help='An evaluation csv file')
    argparser.add_argument('medium', type=str, help='An evaluation csv file')
    argparser.add_argument('large', type=str, help='An evaluation csv file')

    args = argparser.parse_args()

    labels = ['Noisy Baseline', 'Our Preprocessor Medium', 'Our Preprocessor Large']

    csv_noisy = args.noisy
    csv_medium = args.medium
    csv_large = args.large

    df_noisy = pd.read_csv(csv_noisy)
    df_medium = pd.read_csv(csv_medium)
    df_large = pd.read_csv(csv_large)

    grp_snr_noisy = df_noisy.groupby('snr')
    grp_snr_medium = df_medium.groupby('snr')
    grp_snr_large = df_large.groupby('snr')

    grps_snr = [grp_snr_noisy,grp_snr_medium,grp_snr_large]

    grp_nt_noisy = df_noisy.groupby('noise_type')
    grp_nt_medium = df_medium.groupby('noise_type')
    grp_nt_large = df_large.groupby('noise_type')

    grps_nt = [grp_nt_noisy,grp_nt_medium,grp_nt_large]

    tick_labels = []
    x = []
    i = 0
    for key,grp in grp_snr_noisy:
        tick_labels.append(key)
        i += 1
        x.append(i)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    for grps in grps_snr:
        wer_snr = []
        for key,grp in grps:
            wer_snr.append(grp['wer'].mean())
        ax.bar(x,wer_snr,tick_label=tick_labels)
    plt.show()
