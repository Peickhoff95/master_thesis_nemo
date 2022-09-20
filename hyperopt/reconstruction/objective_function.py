import hyperopt
from typing import List, Dict, TypedDict, Type, Any
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

import hyperopt

def train_loss_objective(
    model_config_path: str,
    amount_train_epochs: int,
    model_dir: str,
    params: Dict[str, Any]
):
    config = OmegaConf.load(model_config_path)
    config['trainer']['max_epochs'] = amount_train_epochs
    config['exp_manager']['exp_dir'] = model_dir

    print(params)
    for key, value in params.items():
        traverse_config = config
        sub_keys = key.split('.')
        for sub_key in sub_keys[:-1]:
            traverse_config = traverse_config[sub_key]

        traverse_config[sub_keys[-1]] = value


    #early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=True, check_finite=True)
    #tqdm_progress_bar = TQDMProgressBar(refresh_rate=10)
    #trainer = pl.Trainer(**config.trainer, callbacks=[early_stop_callback,tqdm_progress_bar])
    trainer = pl.Trainer(**config.trainer)
    model = nemo_asr.models.ReconstructionModel(cfg=config.model, trainer=trainer)
    exp_manager(trainer, config.get("exp_manager", None))
    trainer.fit(model)

    loss_dict = trainer.validate(model, verbose=False)[0]
    return {'loss': loss_dict['val_loss'], 'status': hyperopt.STATUS_OK}

