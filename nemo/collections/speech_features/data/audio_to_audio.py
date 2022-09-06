from typing import Optional, Dict, Union

import ipdb

from nemo.core.classes import Dataset
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.common.parts.preprocessing.collections import _Collection, AudioText
import pandas as pd
import collections
from nemo.utils import logging
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType
import torch
from nemo.collections.common import tokenizers

from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor

__all__=[
    'AudioToAudioDataset',
    'AudioToAudioManifestProcessor',
    'AudioAudio'
]

def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig_input, audio_signal_output audio len
    Args:
        batch (Optional[FloatTensor], Optional[FloatTensor],
               LongTensor, LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal_input, audio_signal_output, tokens = [], [], []
    for sig_input, sig_output, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig_input = torch.nn.functional.pad(sig_input, pad)
                sig_output = torch.nn.functional.pad(sig_output, pad)
            audio_signal_input.append(sig_input)
            audio_signal_output.append(sig_output)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal_input = torch.stack(audio_signal_input)
        audio_signal_output = torch.stack(audio_signal_output)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal_input, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal_input, audio_signal_output, audio_lengths, tokens, tokens_lengths

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
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType())
        }

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        preprocessor: 'nemo.collections.asr.modules.audio_preprocessing.AudioToMelSpectrogramPreprocessor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        trim: bool = False,
        use_start_end_token: bool = True,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        self.trim = trim
        self.manifest_processor = AudioToAudioManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            max_duration=max_duration,
            min_duration=min_duration,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
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

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        output = fi, ft, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.manifest_processor.pad_id)

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
        parser,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
    ):
        self.parser = parser

        self.collection = AudioAudio(
            manifest_files=manifest_filepath.split(','),
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_sample(self, sample):
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl

class AudioAudio(_Collection):

    OUTPUT_TYPE = collections.namedtuple('AudioAudioEntity', 'input target duration text text_tokens noise_type snr orig_sr')

    def __init__(self,
                 manifest_files,
                 parser,
                 min_duration: Optional[float]=0,
                 max_duration: Optional[float]=None
                ) -> None:
        output_type = self.OUTPUT_TYPE

        df = pd.DataFrame()
        for manifest_file in manifest_files:
            df = pd.concat([df,pd.read_json(manifest_file, lines=True)])

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

            if row['text'] != '':
                row['text_tokens'] = parser(row['text'])
            else:
                row['text_tokens'] = []

            row['orig_sr'] = 16000
            data.append(output_type(
                row['input'],
                row['target'],
                row['duration'],
                row['text'],
                row['text_tokens'],
                row['noise_type'],
                row['snr'],
                row['orig_sr']
            ))

        super().__init__(data)
