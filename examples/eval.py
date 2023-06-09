import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    config = OmegaConf.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_asr_homepc.yml')
    config.trainer.progress_bar_refresh_rate = 10

    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model, trainer=trainer)
    model.load_state_dict(torch.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small.pt'))
    #exp_manager(trainer, config.get("exp_manager", None))
    __import__('ipdb').set_trace()

    trainer.test(model)


