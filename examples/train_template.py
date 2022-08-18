import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    conf_path = '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small_homepc.yml'
    config = OmegaConf.load(conf_path)
    config.trainer.progress_bar_refresh_rate = 10

    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.ReconstructionModel(cfg=config.model, trainer=trainer)
    exp_manager(trainer, config.get("exp_manager", None))
    trainer.fit(model)
    #__import__('ipdb').set_trace()
