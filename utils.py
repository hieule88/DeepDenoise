import librosa as lib
import torch
import numpy as np
import os
from dc_crn import DCCRN
import torch.nn.functional as F

# Pre-process

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

max_len = 5*16000

def padding(noisy_wav):
    res = max_len - noisy_wav.shape[0]
    noisy_wav_padding = F.pad(input=noisy_wav, pad=(0,res), mode='constant', value=0)
    return noisy_wav_padding

def _preprocess(file_wav):
    wav = load_wav(file_wav)

    batch = []
    miniWav = []

    # chia do dai cho 5*16000, duoc so luong cac doan nho do dai bang 5s
    num_miniWav = wav.shape[0] // max_len + 1
    # gop 8 cai 1 de dua vao mang
    num_batch = num_miniWav // 8 + 1
    # padding 0 vec to fill up batch
    res = num_miniWav % 8 
    padding_batch = torch.zeros(max_len)
    for i in range(res):
        miniWav.append(padding_batch.unsqueeze(0))

    if num_miniWav > 1 :
        for j in range(num_miniWav-1):
            miniWav.append(wav[j*max_len : (j+1)*max_len].unsqueeze(0)) 
    need_add = wav[(num_miniWav-1)*max_len:]
    miniWav.append(padding(need_add).unsqueeze(0))

    for i in range(num_batch):
        tmp_1 = torch.cat((miniWav[i*8+0],miniWav[i*8+1]))
        tmp_2 = torch.cat((miniWav[i*8+2],miniWav[i*8+3]))
        tmp_3 = torch.cat((miniWav[i*8+4],miniWav[i*8+5]))
        tmp_4 = torch.cat((miniWav[i*8+6],miniWav[i*8+7]))
        tmp12 = torch.cat((tmp_1,tmp_2))
        tmp34 = torch.cat((tmp_3,tmp_4))
        batch.append(torch.cat((tmp12,tmp34)))

    return batch

def _load_model():
    model_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/logs"
    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256], batch_size= 8)
    checkpoint = torch.load(os.path.join(model_path, 'parameter_epoch14_2021-04-15 08-03-39.pth' ))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



