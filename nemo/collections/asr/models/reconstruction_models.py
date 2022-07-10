import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import ipdb
from matplotlib import pyplot as plt
from torch.utils.tensorboard._utils import figure_to_image
import numpy as np

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from pytorch_lightning import Trainer

from nemo.collections.asr.parts.preprocessing import process_augmentations
from nemo.core.classes import ModelPT
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel

from nemo.collections.asr.data import audio_to_text_dataset, audio_to_audio_dataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging

from nemo.collections.tts.losses.stftlosses import LogSTFTMagnitudeLoss
from torch.nn.functional import l1_loss

from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig, OmegaConf, open_dict

class ReconstructionMixin(ABC):
    """The ReconstructionMixin is a mixin for classes that are
    tasked to either reconstruct the input or to construct a cleaned form
    based from an encoder model.
    """

    @abstractmethod
    def reconstruct(self, path2audio_files: List[str], batch_size: int = 1):
        pass

class ReconstructionModel(ExportableEncDecModel, ModelPT, ReconstructionMixin):

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def __init__(self, cfg: DictConfig, trainer:Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = ReconstructionModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = ReconstructionModel.from_config_dict(self._cfg.encoder)
        self.decoder = ReconstructionModel.from_config_dict(self._cfg.decoder)

        self.loss = l1_loss

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        if hasattr(self._cfg, 'load_conformer_weights') and self._cfg.load_conformer_weights is not None:
            self.load_state_dict(
                torch.load(
                    self._cfg.load_conformer_weights,
                    map_location=torch.device(self.device)),
                strict=False,
            )
            print("Pre-trained Conformer weights loaded")

        if hasattr(self._cfg, 'freeze_conformer') and self._cfg.freeze_conformer is not None and self._cfg.freeze_conformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Conformer weights frozen")

        if hasattr(self._cfg,'freeze_wsum') and self._cfg.freeze_wsum is not None and self._cfg.freeze_wsum:
            for param in self.encoder.weighted_sum.parameters():
                param.requires_grad = False
            print("Weighted Sum weights frozen")
        else:
            for param in self.encoder.weighted_sum.parameters():
                param.requires_grad = True

        # Setup optional Optimization flags
        #self.setup_optimization_flags()

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_audio_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_audio_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='preprocessor')
        #audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        preprocessor = ReconstructionModel.from_config_dict(config.preprocessor)

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                    'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            #dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)
            dataset = audio_to_audio_dataset.get_audio_dataset(config=config, augmentor=augmentor, preprocessor=preprocessor)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """
                Sets up the training data loader via a Dict-like object.

                Args:
                    train_data_config: A config that contains the information regarding construction
                        of an ASR Training dataset.

                Supported Datasets:
                    -   :class:`~nemo.collections.reconstruction.data.audio_to_audio.AudioToAudioDataset`
                """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
                Sets up the validation data loader via a Dict-like object.

                Args:
                    val_data_config: A config that contains the information regarding construction
                        of an ASR Training dataset.

                Supported Datasets:
                    -   :class:`~nemo.collections.reconstruction.data.audio_to_audio.AudioToAudioDataset`
                """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """
                Sets up the test data loader via a Dict-like object.

                Args:
                    test_data_config: A config that contains the information regarding construction
                        of an ASR Training dataset.

                Supported Datasets:
                    -   :class:`~nemo.collections.reconstruction.data.audio_to_audio.AudioToAudioDataset`
                """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)


    @typecheck()
    def forward(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        #print(f"input shape: {input_signal.shape}")

        #if not has_processed_signal:
        #    processed_signal, processed_signal_length = self.preprocessor(
        #        input_signal=input_signal, length=input_signal_length,
        #    )

        processed_signal = input_signal
        processed_signal_length = input_signal_length
        #print(f"preprocessed shape: {processed_signal.shape}")

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        #print(f"encoded shape: {encoded.shape}")
        log_probs = self.decoder(encoder_output=encoded)
        #print(f"decoded shape: {log_probs.shape}")
        #greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len

    def reconstruct(self, path2audio_files: List[str], batch_size: int = 1):
        pass


    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, target, signal_len = batch
        prediction, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

        if target.shape[2] != prediction.shape[2]:
            target = torch.nn.functional.pad(target, (0, prediction.shape[2]-target.shape[2]), 'constant', 0.0)

        mask = torch.zeros(target.shape).to(self.device)
        for e,e_len in enumerate(encoded_len):
            mask[e,:e_len,:] = 1
        prediction = prediction * mask
        loss_value = self.loss(
            prediction, target, reduction='mean'
        )
        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, target, signal_len = batch
        prediction, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

        if target.shape[2] != prediction.shape[2]:
            target = torch.nn.functional.pad(target, (0, prediction.shape[2]-target.shape[2]), 'constant', 0.0)

        loss_value = self.loss(
            prediction, target, reduction='sum'
        )

        plt.switch_backend('agg')
        fig_spec = plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('Noisy')
        plt.pcolormesh(signal[0].cpu().detach().numpy())
        plt.colorbar()
        plt.subplot(3, 1, 2)
        plt.title('Prediction')
        plt.pcolormesh(prediction[0].cpu().detach().numpy())
        plt.colorbar()
        plt.subplot(3, 1, 3)
        plt.title('Clean')
        plt.pcolormesh(target[0].cpu().detach().numpy())
        plt.colorbar()
        fig_spec = figure_to_image(fig_spec, close=True)

        return {
            'val_loss': loss_value,
            'spectograms': fig_spec,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_loss': logs['val_loss'],
        }
        return test_logs

#    def validation_epoch_end(self, outputs, dataloader_idx: int = 0):
#        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
#        specs = [x['spectograms'] for x in outputs[:10]]
#        tensorboard_logs = {'val_loss': val_loss_mean}
#        #Log spectograms
#        self.trainer.logger.experiment.add_image('spectograms', np.stack(specs), global_step=self.global_step, dataformats='NCHW')
#        #Log weighted sum
#        plt.switch_backend('agg')
#        fig_wsum = plt.figure()
#        plt.bar(range(self.encoder.weighted_sum.weight.shape[1]), self.encoder.weighted_sum.weight[0].cpu().detach().numpy())
#        self.trainer.logger.experiment.add_image('weighted_sum/weights_bar', figure_to_image(fig_wsum, close=True), global_step=self.global_step)
#        self.trainer.logger.experiment.add_histogram('weighted_sum/weights', self.encoder.weighted_sum.weight, global_step=self.global_step)
#        if self._cfg.encoder.wsum_bias:
#            self.trainer.logger.experiment.add_histogram('weighted_sum/bias', self.encoder.weighted_sum.bias, global_step=self.global_step)
#        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}


    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        specs = [x['spectograms'] for x in outputs[:10]]
        tensorboard_logs = {'val_loss': val_loss_mean}
        #Log spectograms
        self.trainer.logger.experiment.add_image('spectograms', np.stack(specs), global_step=self.global_step, dataformats='NCHW')
        #Log weighted sum
        plt.switch_backend('agg')
        fig_wsum = plt.figure()
        plt.bar(range(self.encoder.weighted_sum.weight.shape[1]), self.encoder.weighted_sum.weight[0].cpu().detach().numpy())
        self.trainer.logger.experiment.add_image('weighted_sum/weights_bar', figure_to_image(fig_wsum, close=True), global_step=self.global_step)
        self.trainer.logger.experiment.add_histogram('weighted_sum/weights', self.encoder.weighted_sum.weight, global_step=self.global_step)
        if self._cfg.encoder.wsum_bias:
            self.trainer.logger.experiment.add_histogram('weighted_sum/bias', self.encoder.weighted_sum.bias, global_step=self.global_step)
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': val_loss_mean}
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}

    ############################################################################
    #Delete Later
    @torch.no_grad()
    def transcribe(
            self,
            paths2audio_files: List[str],
            batch_size: int = 4,
            logprobs: bool = False,
            return_hypotheses: bool = False,
            num_workers: int = 0,
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                }

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx][: logits_len[idx]]
                            hypotheses.append(lg.cpu().numpy())
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': {},
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
