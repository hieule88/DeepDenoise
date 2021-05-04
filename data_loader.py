from torch.utils.data import Dataset
import librosa as lib
import os
import numpy as np
import torch
import train_utils
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
import torchaudio


def load_wav(path):
    signal, _ = torchaudio.load(path)
    signal = signal.reshape(-1)
    return signal

class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, noise_paths, max_time, loader=load_wav):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.noise_paths = noise_paths
        self.loader = loader
        self.max_len = max_time*16000

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        
        pding = self.padding(self.loader(noisy_file), self.loader(clean_file))
        return pding

    def __len__(self):
        return len(self.noisy_paths)

    def padding(self, noisy_wav, clean_wav):
        if (noisy_wav.shape[0] < self.max_len ):
            noisy_wav_padding = self.loader(self.noise_paths[int(random.uniform(0, len(self.noise_paths)-1))])[:self.max_len]
            noisy_wav_padding[:noisy_wav.shape[0]] = noisy_wav

            res = self.max_len - noisy_wav.shape[0]
            clean_wav_padding = F.pad(input=clean_wav, pad=(0,res), mode='constant', value=0)
            return noisy_wav_padding, clean_wav_padding
        else :
            return noisy_wav[:self.max_len], clean_wav[:self.max_len]

