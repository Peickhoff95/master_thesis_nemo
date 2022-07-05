from typing import Optional, Dict, Union

import ipdb

from nemo.core.classes import Dataset
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.common.parts.preprocessing.collections import _Collection
import pandas as pd
import collections
from nemo.utils import logging
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch

from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor

__all__=[
    'AudioToAudioDataset',
    'AudioToAudioManifestProcessor',
    'AudioAudio'
]

def _speech_collate_fn(batch):
    """collate batch of audio sig_input, audio_signal_output audio len
    Args:
        batch (Optional[FloatTensor], Optional[FloatTensor],
               LongTensor, LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, _, audio_lengths  = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal_input, audio_signal_output = [], []
    for sig_input, sig_output, sig_len, in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig_input = torch.nn.functional.pad(sig_input, pad)
                sig_output = torch.nn.functional.pad(sig_output, pad)
            audio_signal_input.append(sig_input)
            audio_signal_output.append(sig_output)

    if has_audio:
        audio_signal_input = torch.stack(audio_signal_input)
        audio_signal_output = torch.stack(audio_signal_output)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal_input, audio_lengths = None, None

    return audio_signal_input, audio_signal_output, audio_lengths,

class AudioToAudioDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "duration": 0.82}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_input_signal': NeuralType(('B', 'D', 'T'), AudioSignal()),
            'audio_output_signal': NeuralType(('B', 'D', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType())
            
        }

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        preprocessor: 'nemo.collections.asr.modules.audio_preprocessing.AudioToMelSpectrogramPreprocessor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        trim: bool = False
    ):
        self.trim = trim
        self.manifest_processor = AudioToAudioManifestProcessor(
            manifest_filepath=manifest_filepath,
            max_duration=max_duration,
            min_duration=min_duration,
        )

        self.featurizer = WaveformFeaturizer(
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
        )

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = None

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]

        features_input = self.featurizer.process(
            sample.input,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr
        )

        features_target = self.featurizer.process(
            sample.target,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr
        )

        fi, ft, fl = features_input, features_target, torch.tensor(features_input.shape[0]).long()
        fi = fi[None, :]
        ft = ft[None, :]
        fl = fl[None]
        if self.preprocessor is not None:
            processed_input_signal, processed_signal_length = self.preprocessor(
                input_signal=fi, length=fl,
            )
            processed_target_signal, _ = self.preprocessor(
                input_signal=ft, length=fl,
            )
            fi, ft, fl = processed_input_signal[0], processed_target_signal[0], processed_signal_length[0]

        output = fi, ft, fl,

        return output

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch)

class AudioToAudioManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"input": "/path/to/audio.wav", "output": "/path/to/audio.wav", 
        "duration": 23.147}
    {"input": "/path/to/audio.wav", "output": "/path/to/audio.wav",
        "duration": 0.82}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
    """

    def __init__(
        self,
        manifest_filepath: str,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ):

        self.collection = AudioAudio(
            manifest_files=manifest_filepath.split(','),
            min_duration=min_duration,
            max_duration=max_duration,
        )

class AudioAudio(_Collection):

    OUTPUT_TYPE = collections.namedtuple('AudioAudioEntity', 'input target duration offset speaker orig_sr')

    def __init__(self,
                 manifest_files,
                 min_duration: Optional[float]=0,
                 max_duration: Optional[float]=None
                ) -> None:
        output_type = self.OUTPUT_TYPE

        df = pd.DataFrame()
        for manifest_file in manifest_files:
            df = df.append(pd.read_json(manifest_file, lines=True))

        optional_columns = ['text']
        for optional_column in optional_columns:
            if optional_column not in df.columns:
                df[optional_column] = None
        
        filtered_min_duration = df['duration'] < min_duration
        filtered_max_duration = df['duration'] > max_duration\
                if max_duration != None\
                else filtered_min_duration

        filtered_duration = df.loc[filtered_max_duration | filtered_min_duration, 'duration'].sum()
        logging.info(f'We filtered {filtered_duration} seconds of samples')
        df = df[~(filtered_max_duration | filtered_min_duration)] #type: pd.DataFrame

        data = []
        for _, row in df.iterrows():
            data.append(output_type(
                row['input'],
                row['target'],
                row['duration'],
                row['text'],
                row['noise_type'],
                row['snr']
            ))

        super().__init__(data)
