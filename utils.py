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

def _process(batch, model):

    with torch.no_grad():
        denoise_flt = model(batch).to('cpu')

    return denoise_flt

def _load_model(model_path):
    global batch_size
    # model_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/logs"
    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256], batch_size= batch_size)
    checkpoint = torch.load(os.path.join(model_path, 'parameter_epoch14_2021-04-15 08-03-39.pth' ))
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


