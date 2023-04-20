import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
import re
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser
import numpy as np

if __name__ == '__main__':

    
    argparser = ArgumentParser()
    argparser.add_argument('config', type=str, help='A config yaml file, like denoise_example.yml')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    exp_dir = config.expdir_path
    manifest_paths = config.manifest_path
    rec_config_path = config.reconstruction.config_path
    rec_ckpt_path = config.reconstruction.checkpoint_path
   
    device = torch.device('cuda:0')

    for manifest_path in manifest_paths.split(','):

        df = pd.read_json(manifest_path, lines=True)

        rec_config = OmegaConf.load(rec_config_path)
        rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
        chkpt = torch.load(rec_ckpt_path, map_location=device)

        rec_model.load_state_dict(chkpt['state_dict'])
        rec_model.to(device)
        denoised_specs = rec_model.reconstruct(df['input'])
        noisy_specs = rec_model.return_spectogram(df['input'])
        clean_specs = rec_model.return_spectogram(df['target'])

        del rec_model

        mae_noisy_clean = []
        mae_noisy_denoised = []
        mae_denoised_clean = []

        for denoised_spec,noisy_spec,clean_spec in zip(tqdm(denoised_specs, desc='Transcribing denoised'),noisy_specs,clean_specs):
            
            mae_noisy_clean.append(np.sum(np.abs((noisy_spec - clean_spec))) / np.sum(clean_spec.shape)) 
            mae_noisy_denoised.append(np.sum(np.abs((noisy_spec - denoised_spec))) / np.sum(denoised_spec.shape)) 
            mae_denoised_clean.append(np.sum(np.abs((denoised_spec - clean_spec))) / np.sum(clean_spec.shape)) 

        df['mae_noisy_clean'] = mae_noisy_clean
        df['mae_noisy_denoised'] = mae_noisy_denoised
        df['mae_denoised_clean'] = mae_denoised_clean

        df.to_csv(exp_dir + manifest_path.split('/')[-1].split('_txt')[0]  + '_eval_mae.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
