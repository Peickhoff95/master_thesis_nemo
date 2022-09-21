import os
import argparse
import torch
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import numpy as np
from numpy import typing as npt
from typing import List
import pandas as pd
from tqdm import tqdm 

def denoise_audiofiles(audiofiles_paths: List[str], conf_path: str, chkpt_path: str, out_dir: str, batch_size: int = 4, num_workers: int = 0)->npt.NDArray:
    """TODO: Docstring for denoise_audiofile.

                    :arg1: TODO
                    :returns: TODO

                    """
    rec_config = OmegaConf.load(conf_path)
    rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
    chkpt = torch.load(chkpt_path)
    rec_model.load_state_dict(chkpt['state_dict'])

    rec_model.cuda()
    rec_model.eval()
    
    outfile_paths = []
    for audiofile_path in tqdm(audiofiles_paths):
        outfile_path = os.path.join(out_dir,audiofile_path.split('/')[-1].split('.')[0] + '.npy')
        outfile_paths.append(outfile_path)
        denoised_logspec = rec_model.reconstruct([audiofile_path], batch_size=batch_size, num_workers=num_workers, verbose=False)    
        if len(denoised_logspec) > 1: print(len(denoised_logspec))
        np.save(outfile_path, denoised_logspec[0])
    
    return outfile_paths 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Resample wav files')
    parser.add_argument('CONF',  type=str, help='Config YAML for Reconstruction Model')
    parser.add_argument('CHKPT',  type=str, help='Checkpoint file for Reconstruction Model')
    parser.add_argument('MANIFEST',  type=str, help='Manifest file path of dataset') 
    parser.add_argument('OUTDIR',  type=str, help='Output directory for denoised spectogramms')

    parser.add_argument('--batch_size', metavar='batch_size',default=4, type=int, help='batchsize for denoising function')
    parser.add_argument('--num_workers', metavar='num_workers',default=0, type=int, help='number of workers for loading file')
    args = parser.parse_args()
    conf_path = args.CONF
    chkpt_path = args.CHKPT
    manifestfile_path = args.MANIFEST
    out_dir = args.OUTDIR
    batch_size = args.batch_size
    num_workers = args.num_workers
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    manifest_df = pd.read_json(manifestfile_path, lines=True)
    outfile_paths = denoise_audiofiles(
        manifest_df['input'],
        conf_path=conf_path, 
        chkpt_path=chkpt_path,
        out_dir=out_dir,
        batch_size=batch_size, 
        num_workers=num_workers,
        )
    manifest_df['input'] = outfile_paths
    
    manifest_outpath = os.path.join(out_dir,manifestfile_path.split('/')[-1])
    manifest_df.to_json(manifest_outpath, orient='records', lines=True)    
    print(f'manifest saved to: {manifest_outpath}')

