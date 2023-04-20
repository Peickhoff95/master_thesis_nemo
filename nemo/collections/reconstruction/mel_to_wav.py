import numpy as np
import torch
import torch.nn as nn
import torchaudio

class dB_to_Amplitude(nn.Module):
    def __call__(self, features):
        return(torch.from_numpy(np.power(10, features.numpy()/10.0)))

def get_waveform_from_logMel(features, n_fft=400, hop_length=160, sr=16000):
    print(features.shape)
    n_mels = features.shape[-2]
    inverse_transform = torch.nn.Sequential(
           # dB_to_Amplitude(),
            torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=n_mels, sample_rate=sr),
            torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)
            )
    waveform = inverse_transform(torch.squeeze(features))
    return torch.unsqueeze(waveform, 0)
