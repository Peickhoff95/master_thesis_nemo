import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

if __name__ == "__main__":

    device = 'cuda:0'
    argparser = ArgumentParser()
    argparser.add_argument('config_yml', type=str, help='A run configuration yaml file')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config_yml)
    config.trainer.progress_bar_refresh_rate = 10

    chkpt_path = '/home/eickhoff/data/4eickhof/repos/master_thesis_nemo/experiments/Conformer-Reconstruction-Large-Hyperopt-Nosched/2022-09-13_00-27-38/checkpoints/Conformer-Reconstruction-Large-Hyperopt-Nosched--val_loss=0.1690-epoch=95.ckpt'
    chkpt = torch.load(chkpt_path, map_location=device)
    
    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.ReconstructionWhisperModel(cfg=config.model, trainer=trainer)
    model.load_state_dict(chkpt['state_dict'], strict=False) 
   # __import__('ipdb').set_trace()
    exp_manager(trainer, config.get("exp_manager", None))
    trainer.fit(model)

