import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('config_yml', type=str, help='A run configuration yaml file')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config_yml)
    config.trainer.progress_bar_refresh_rate = 10

    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model, trainer=trainer)
    exp_manager(trainer, config.get("exp_manager", None))
    trainer.fit(model)
    #__import__('ipdb').set_trace()
