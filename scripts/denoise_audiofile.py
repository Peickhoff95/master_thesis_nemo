import os
import argparse
import torch
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import numpy as np
from numpy import typing as npt

def denoise_audiofile(conf_path: str, chkpt_path: str, audiofile_path: str, batch_size: int = 4, num_workers: int = 0)->npt.NDArray:
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
    
    denoised_logspec = rec_model.reconstruct([audiofile_path], batch_size=batch_size, num_workers=num_workers)
    
    return np.stack(denoised_logspec, axis=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Resample wav files')
    parser.add_argument('CONF',  type=str, help='Config YAML for Reconstruction Model')
    parser.add_argument('CHKPT',  type=str, help='Checkpoint file for Reconstruction Model')
    parser.add_argument('FILE',  type=str, help='Input wav file to be downsampled')
    parser.add_argument('--out_file','-o', metavar='output_file', default=None, type=str, help='output file. if none specified, target file equals input file')
    
    parser.add_argument('--batch_size', metavar='batch_size',default=4, type=int, help='batchsize for denoising function')
    parser.add_argument('--num_workers', metavar='num_workers',default=0, type=int, help='number of workers for loading file')
    args = parser.parse_args()
    conf_path = args.CONF
    chkpt_path = args.CHKPT
    audiofile_path = args.FILE
    batch_size = args.batch_size
    num_workers = args.num_workers
    outfile_path = args.out_file if not args.out_file is None else args.FILE.split('.')[0] + '.npy'
    
    denoised_logspec = denoise_audiofile(conf_path=conf_path, chkpt_path=chkpt_path, audiofile_path=audiofile_path, batch_size=batch_size, num_workers=num_workers)
    np.save(outfile_path, denoised_logspec)
    print(f'Log-Mel Spectograms saved to {outfile_path}')

