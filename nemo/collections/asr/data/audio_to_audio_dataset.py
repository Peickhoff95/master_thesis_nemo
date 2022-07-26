import json
import random
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig, open_dict
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text, audio_to_text_dali
from nemo.collections.speech_features.data import audio_to_audio
from nemo.utils import logging


def inject_dataloader_value_from_model_config(model_cfg: dict, dataloader_cfg: DictConfig, key: str):
    """
    Extracts the label set provided at the top level of the model, and propagates it to the dataloader
    config.

    Args:
        model_cfg: A DictConfig representing the model's config.
        dataloader_cfg: A DictConfig representing the individual data loader
        key: A str value representing a key in the model_cfg whose value will be propagated to the
            dataloader config.
    """
    if key not in model_cfg:
        logging.info(
            f"Model level config does not container `{key}`, please explicitly provide `{key}` to the dataloaders."
        )
        return

    if not isinstance(dataloader_cfg, DictConfig):
        dataloader_cfg = DictConfig(dataloader_cfg)

    # If key exists in the data loader config (either set explicitly or as a placeholder (via None))
    if key in dataloader_cfg:
        # Dataloader `labels` is provided and is non-null
        if dataloader_cfg[key] is not None and model_cfg[key] != dataloader_cfg[key]:
            # Model level `labels` dont match Dataloader level `labels`
            logging.warning(
                f'`{key}` is explicitly provided to the data loader, and is different from '
                f'the `{key}` provided at the model level config.\n'
                f'If this is incorrect, please set the dataloader\'s `{key}` to None.'
            )

        else:
            # Dataloader `key` is None or values match
            # Propagate from model level `key` (even if they match)
            with open_dict(dataloader_cfg):
                dataloader_cfg[key] = model_cfg[key]

    else:
        # If key key doesnt even exist in dataloader_cfg, inject it explicitly
        with open_dict(dataloader_cfg):
            dataloader_cfg[key] = model_cfg[key]

def get_audio_dataset(
        config: dict,
        tokenizer,
        augmentor: Optional['AudioAugmentor'] = None,
        preprocessor: Optional['AudioPreprocessor'] = None) -> audio_to_audio.AudioToAudioDataset:
    """
    Instantiates an AudioToAudioDataset.

    Args:
        config: Config of the AudioToAudioDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
        preprocessor Optional AudioPreprocessor object for preprocessing audio features.

    Returns:
        An instance of AudioToAudioDataset.
    """

    dataset = audio_to_audio.AudioToAudioDataset(
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        preprocessor=preprocessor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        trim=config.get('trim_silence', False),
    )
    return dataset
