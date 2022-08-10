import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__ == "__main__":
    config = OmegaConf.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_joint_asr_homepc.yml')
    config.trainer.progress_bar_refresh_rate = 10

    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.ReconstructionASRModel(cfg=config.models, trainer=trainer)
    
    chkpt = torch.load('/home/patrick/Projects/master_thesis_nemo/experiments/Conformer-Reconstruction/2022-07-06_15-57-08/checkpoints/Conformer-Reconstruction--val_loss=195796.8438-epoch=93.ckpt')
    model.load_state_dict(chkpt['state_dict'], strict=False)

    #__import__('ipdb').set_trace()
    exp_manager(trainer, config.get("exp_manager", None))
    trainer.fit(model)
    #__import__('ipdb').set_trace()
