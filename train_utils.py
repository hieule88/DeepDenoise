import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pesq import pesq
import os
import gc
import sys
import time
from callbacks import EarlyStopping, LRScheduler

def get_all_names(train_test, dns_home):
    train_names = train_test["train"]
    test_names = train_test["test"]

    train_noisy_names = []
    train_clean_names = []
    test_noisy_names = []
    test_clean_names = []

    for name in train_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_home, 'clean_train_youtube', name)
        noisy_file = os.path.join(dns_home, 'noisy_train_youtube', name)
        train_clean_names.append(clean_file)
        train_noisy_names.append(noisy_file)
    for name in test_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_home, 'clean_train_youtube', name)
        noisy_file = os.path.join(dns_home, 'noisy_train_youtube', name)
        test_clean_names.append(clean_file)
        test_noisy_names.append(noisy_file)
    return train_noisy_names, train_clean_names, test_noisy_names, test_clean_names


def test_epoch(model, test_iter, device, criterion, test_all=False):
    
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        i = 0
        for ind, (x, y) in enumerate(test_iter):
            x = x.to(device).float()
            y = y.to(device).float()
            y_p = model(x)[1]
            loss = criterion(source = y , estimate_source=  y_p)
            loss_sum += loss
            i+=1

            if not test_all:
                break
    return loss_sum / i


def train(model, optimizer, criterion, train_iter, test_iter, max_epoch, device, log_path, just_test=False):
    train_losses = []
    test_losses = []
    for epoch in range(max_epoch):
        loss_sum = 0
        i = 0
        for step, (x, y) in enumerate(train_iter):
            x = x.to(device).float()
            y = y.to(device).float()

            with torch.autograd.detect_anomaly():
                model.train()
                optimizer.zero_grad()
                y_p = model(x)[1]
                loss = criterion(estimate_source = y_p, source = y)
                if step == 0 and epoch == 0:
                    loss.backward()
                    loss_sum += loss
                    i += 1
                    test_loss = test_epoch(model, test_iter, device, criterion, test_all=False)
                    print(
                        "first test step:%d,train loss:%.5f,test loss:%.5f" % (
                            step, loss_sum / i, test_loss)
                    )
                    train_losses.append(loss_sum.cpu().detach().numpy() / i)
                    test_losses.append(test_loss.cpu().detach().numpy())
                else:
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss
                    i += 1
                if step % int(len(train_iter) // 10) == 0 or step == len(train_iter) - 1:
                    test_loss = test_epoch(model, test_iter, device, criterion, test_all=False)
                    print(
                        "epoch:%d,step:%d,train loss:%.5f,test loss:%.5f,time:%s" % (
                            epoch, step, loss_sum / i, test_loss, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
                        )
                    )
                    train_losses.append(loss_sum.cpu().detach().numpy() / i)
                    test_losses.append(test_loss.cpu().detach().numpy())
                    plt.plot(train_losses)
                    plt.plot(test_losses)
                    plt.savefig(os.path.join(log_path, "loss_time%s_epoch%d_step%d.png" % (
                        time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch, step)), dpi=150)
                    plt.show()
                
                if (step % int(len(train_iter) // 3) == 0 and step != 0) or step == len(train_iter) - 1:
                    print("save model,epoch:%d,step:%d" % (epoch, step))
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_sum
                                },
                                os.path.join(log_path, "parameter_epoch%d_%s.pth" % (epoch, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))))
                    pickle.dump({"train loss": train_losses, "test loss": test_losses},
                                open(os.path.join(log_path, "loss_time%s_epoch%d.log" % (
                                    time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch)), "wb"))

                if just_test:
                    break


def get_train_test_name(dns_home):
    all_name = []
    for i in os.walk(os.path.join(dns_home, "noisy_train_youtube")):
        for name in i[2]:
            all_name.append(name)
    train_names = all_name[:-len(all_name) // 5]
    test_names = all_name[-len(all_name) // 5:]
    data = {"train": train_names, "test": test_names}
    pickle.dump(data, open("./train_test_names.data", "wb"))
    return data

def get_noisy_name(dns_home):
    with open(os.path.join(dns_home, "noise/noise_used.txt")) as f:
        all_names = f.readlines()
    all_names = [x.strip() for x in all_names] 
    return all_names