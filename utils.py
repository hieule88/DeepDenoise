import librosa as lib
import torch
import numpy as np
import os
from dc_crn import DCCRN
import torch.nn.functional as F
import math
import gc

# Pre-process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_wav(path, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    return torch.tensor(signal)

max_len = 5*16000

def padding(noisy_wav):
    res = max_len - noisy_wav.shape[0]
    noisy_wav_padding = F.pad(input=noisy_wav, pad=(0,res), mode='constant', value=0)
    return noisy_wav_padding

def _process(file_wav, model):
    wav = load_wav(file_wav)
    batch = []
    miniWav = []

    # chia do dai cho 5*16000, duoc so luong cac doan nho do dai bang 5s
    num_miniWav = math.ceil(wav.shape[0] / max_len) 
    # gop 8 cai 1 de dua vao mang
    num_batch = math.ceil(num_miniWav / 8)
    # padding 0 vec to fill up batch
    res = 8 - (num_miniWav % 8) 
    padding_batch = torch.zeros(max_len)

    if num_miniWav > 1 :
        for j in range(num_miniWav-1):
            miniWav.append(wav[j*max_len : (j+1)*max_len].unsqueeze(0)) 
    need_add = wav[(num_miniWav-1)*max_len:]
    miniWav.append(padding(need_add).unsqueeze(0))

    for i in range(res):
        miniWav.append(padding_batch.unsqueeze(0))

    tmp_1 = torch.cat((miniWav[0],miniWav[1]))
    tmp_2 = torch.cat((miniWav[2],miniWav[3]))
    tmp_3 = torch.cat((miniWav[4],miniWav[5]))
    tmp_4 = torch.cat((miniWav[6],miniWav[7]))
    tmp12 = torch.cat((tmp_1,tmp_2))
    tmp34 = torch.cat((tmp_3,tmp_4))
    tmp = torch.cat((tmp12,tmp34)).to(device)
    with torch.no_grad():
        denoise_flt = model(tmp).reshape(1,640000).to('cpu')

    if num_batch > 1:
        for i in range(1,num_batch):
            tmp_1 = torch.cat((miniWav[i*8+0],miniWav[i*8+1]))
            tmp_2 = torch.cat((miniWav[i*8+2],miniWav[i*8+3]))
            tmp_3 = torch.cat((miniWav[i*8+4],miniWav[i*8+5]))
            tmp_4 = torch.cat((miniWav[i*8+6],miniWav[i*8+7]))
            tmp12 = torch.cat((tmp_1,tmp_2))
            tmp34 = torch.cat((tmp_3,tmp_4))
            tmp = torch.cat((tmp12,tmp34)).to(device)
            with torch.no_grad():
                denoise = model(tmp).reshape(1,640000).to('cpu')
            
            denoise_flt = torch.cat((denoise_flt,denoise), -1)

    return denoise_flt

def _load_model(model_path):
    #model_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/logs"
    #model_path = "/home/hieule/DeepDenoise"
    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256], batch_size= 8)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def combine_Out(batch, len_input, model):

    denoise = model(batch[0])[1]
    denoise_ftl = denoise.reshape(1, 640000)
    if len(batch) > 1:

        for i in range(1, len(batch)) :
            
            denoise = model(batch[i])[1].reshape(1,640000)

            denoise_ftl = torch.cat((denoise_ftl,denoise), -1)

    return denoise_ftl[:,:len_input]


