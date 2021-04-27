from pypesq import pesq
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import gc
import sys
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam 
from dc_crn import DCCRN
import torch.nn.functional as F
from tqdm import tqdm

sr = 16000
# get file's name
def get_all_names(test_names, dns_home):

    test_noisy_names = []
    test_clean_names = []

    for name in test_names:
        clean_file = os.path.join(dns_home, 'denoise_test_clean', name)
        noisy_file = os.path.join(dns_home, 'denoise_test_noisy', name)
        test_clean_names.append(clean_file)
        test_noisy_names.append(noisy_file)
    return test_noisy_names, test_clean_names

def get_test_name(dns_home):
    all_name = []
    for i in os.walk(os.path.join(dns_home, "denoise_test_noisy")):
        for name in i[2]:
            all_name.append(name)
    return all_name

# Load data
def load_wav(path):
    signal, _ = torchaudio.load(path)
    signal = signal.reshape(-1)
    return signal

class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, max_time, loader=load_wav):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths

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

            res = self.max_len - noisy_wav.shape[0]
            noisy_wav_padding = F.pad(input=noisy_wav, pad=(0,res), mode='constant', value=0)
            clean_wav_padding = F.pad(input=clean_wav, pad=(0,res), mode='constant', value=0)
            return noisy_wav_padding, clean_wav_padding
        else :
            return noisy_wav[:self.max_len], clean_wav[:self.max_len]

 # load data
batch_size = 8

dns_home = "/data/hieuld/data_test"
model_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/logs"

test = get_test_name(dns_home)
test_noisy_names, test_clean_names = get_all_names(test, dns_home=dns_home)

data_test = WavDataset(test_noisy_names, test_clean_names, 5)
test_dataloader = DataLoader(data_test, batch_size= batch_size, shuffle= False)

def test_epoch(model, test_iter, device, criterion, test_all=False):
    
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        i = 0
        for step, (x, y) in tqdm(enumerate(test_iter), ascii ="123456789$", desc = "Processing"):
            x = x.to(device).float()
            y = y.to(device).float()
            y_p = model(x)[1]
            loss = criterion(step = step , source = y , estimate_source=  y_p)
            loss_sum += loss
            i+=1
            
    return loss_sum / i
    
def loss_pesq(step, source , estimate_source):
    source = source.cpu()
    estimate_source = estimate_source.cpu()
    
    score = 0
    for i in range(source.shape[0]):
        score = score + pesq(source[i], estimate_source[i], sr)
        
        torchaudio.save(os.path.join(dns_home, 'predict_folder',
                        test[step*batch_size + i]), 
                        estimate_source[i].unsqueeze(0), sample_rate = 16000)

    return score / source.shape[0]

if __name__ == '__main__':
# load model

    model = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256], batch_size= batch_size)
    optimizer = Adam(model.parameters())

    checkpoint = torch.load(os.path.join(model_path, 'parameter_epoch12_2021-04-14 22-09-38.pth' ))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_score = test_epoch(model, test_dataloader, device, loss_pesq )
    print(test_score)



