import torch
import torch.nn as nn
import os
import sys
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT 
from torch.optim import Adam 
from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm
import data_loader as loader
from torch.utils.data import DataLoader
import pickle
import train_utils
from si_snr import *
from dc_crn import DCCRN


if __name__ == '__main__':
    root_path = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN/"
    dns_home = "/data/hieuld/data"  # dir of dns-datas
    save_file = root_path + "logs"

    
    train_test = train_utils.get_train_test_name(dns_home)
    train_noisy_names, train_clean_names, test_noisy_names, test_clean_names = \
        train_utils.get_all_names(train_test, dns_home=dns_home)
    noise_names = train_utils.get_noisy_name(dns_home)

    torch.autograd.set_detect_anomaly(True)
    epochs = 200
    batch_size = 8
    lr = 0.001

    data_train = loader.WavDataset(train_noisy_names, train_clean_names, noise_names, 4)
    data_test = loader.WavDataset(test_noisy_names, test_clean_names, noise_names, 4)

    train_dataloader = DataLoader(data_train, batch_size= batch_size, shuffle= True)
    test_dataloader = DataLoader(data_test, batch_size= batch_size, shuffle= True)
    
    #net = DCCRN(batch_size = batch_size, rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
    net = torch.load(os.path.join(save_file, 'parameter_epoch21_2021-04-07 06-18-49.pth'))
    print(net.__dict__)
    optimizer = Adam(net.parameters(), lr= lr, weight_decay = 0.5)
    criterion = SiSnr()

    checkpoint = torch.load(os.path.join(save_file, 'parameter_epoch21_2021-04-07 06-18-49.pth'))
    net.load_state_dict(checkpoint)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net= net.to(device)

    train_utils.train(model=net, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader,
                  test_iter=test_dataloader, max_epoch=500, device=device, log_path=save_file,
                  just_test=False)