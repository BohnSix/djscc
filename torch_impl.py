import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import json
import time
import torch
import pickle
import datetime
import matplotlib
import torchvision
import numpy as np
from torch import nn
from glob import glob
from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter



class Encoder(nn.Module):
    def __init__(self, conv_depth):
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.PReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(32, conv_depth, 5, 1, 2),
            nn.PReLU(),
        ])

    def forward(self, x):
        for layer in self.sublayers:
            x = layer(x)
        # return x.type(torch.complex64)
        return torch.complex(x, torch.zeros_like(x))

class Decoder(nn.Module):
    def __init__(self, conv_depth):
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.ConvTranspose2d(conv_depth, 32, 5, 1, 2),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, 5, 1, 2),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, 5, 1, 2),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 16, 5, 2, 2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 3, 5, 2, 2, output_padding=1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        x = torch.real(x).float()
        for layer in self.sublayers:
            x = layer(x)
        return x

class Channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super().__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr
    
    def awgn(self, x, stddev, lst):
        cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(lst[0],lst[1],lst[2]*2)).view(np.complex128)
        cmplx_dist = torch.from_numpy(cmplx_dist)
        noise = cmplx_dist.cuda() * stddev
        return x + noise

    def forward(self, channal_input, lst):
        # print("channel_snr: {}".format(self.channel_snr))
        k = np.prod(lst)
        snr = 10**(self.channel_snr/10.0)
        abs_val = torch.abs(channal_input)
        signl_pwr = torch.sum(torch.square(abs_val)) / k
        noise_pwr = signl_pwr / snr
        noise_stddev = torch.sqrt(noise_pwr)

        if self.channel_type == "awgn":
            channal_output = self.awgn(channal_input, noise_stddev, lst)
            h = torch.ones_like(channal_input)

        return channal_output, h

class JSCC(nn.Module):
    def __init__(self, conv_depth, snr_db=10):
        super().__init__()
        self.encoder = Encoder(conv_depth)
        self.channel = Channel("awgn", snr_db)
        self.decoder = Decoder(conv_depth)

    def forward(self, inputs, snr_db=10, P=1):
        prev_chn_gain = None
        chn_in = self.encoder(inputs)

        lst = list(chn_in.shape)[1:]
        chn_out, h = self.channel(chn_in, lst)

        decoded_img = self.decoder(chn_out)

        return decoded_img, chn_out

def Calculate_filters(comp_ratio, F=8, n=3072):
    K = (comp_ratio*n)/F**2
    return int(K)

# ###############################################################
# compression_ratios = [0.04, 0.09, 0.17, 0.25, 0.33, 0.42, 0.49]
# filter_size = []
# for comp_ratio in compression_ratios:
#     K = Calculate_filters(comp_ratio)
#     filter_size.append(K)

# print(filter_size)  # [2, 4, 8, 12, 16, 20, 24]
# ###############################################################

SNR = 10
COMPRESSION_RATIO = 0.17
K = Calculate_filters(COMPRESSION_RATIO)

"""
rm checkpoints/*
rm -r train_logs/*
rm validation_imgs/*

nohup python -u torch_impl.py > train_logs/snr10_c17.log 2>&1 &
"""

EPOCHS = 2500
NUM_WORKERS = 4
LEARNING_RATE = 0.001
CHANNEL_TYPE = "awgn"
CHANNEL_SNR_TRAIN = 10
TRAIN_IMAGE_NUM = 50000
TEST_IMAGE_NUM = 10000
TRAIN_BS = 8192
TEST_BS = 4096

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BS, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS, shuffle=False, num_workers=NUM_WORKERS)

image_dim = 32 * 32 * 3

model = JSCC(K, snr_db=SNR).cuda()

# model.load_state_dict(torch.load("/media/bohnsix/djscc/checkpoints/jscc_model_17"))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()
        optimizer.zero_grad()
        decoded_img, chn_out = model(inputs)
       
        loss = loss_fn(decoded_img, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(trainloader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return running_loss

writer = SummaryWriter(f'train_logs/deepjscc_{CHANNEL_TYPE}_snr{SNR}_c{COMPRESSION_RATIO}')

best_vloss = 1.
change_lr_flag = True

for epoch in range(1, EPOCHS+1):
    if epoch > 640 and change_lr_flag:
        LEARNING_RATE = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        change_lr_flag = False
        print("Update LR to 1e-4")

    cur = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'EPOCH {epoch} starts at {cur}')

    model.train(True)
    avg_loss = train_one_epoch(epoch, writer) * 1e4 / TRAIN_IMAGE_NUM

    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for _ in range(10):
            for i, vdata in enumerate(testloader):
                vinputs, vlabels = vdata
                vinputs = vinputs.cuda()
                decoded_img, chn_out = model(vinputs)
                vloss = loss_fn(decoded_img, vinputs)

                running_vloss += vloss

        a = vinputs[:256].detach().cpu().numpy().reshape(16, 16, 3, 32, 32).transpose(0, 1, 3, 4, 2)
        b = decoded_img[:256].detach().cpu().numpy().reshape(16, 16, 3, 32, 32).transpose(0, 1, 3, 4, 2)
        c = (np.hstack(np.hstack(np.concatenate([a, b], 3)))[..., ::-1] * 255).astype(np.uint8)
        cv2.imwrite(f"validation_imgs/validation_snr{SNR}_c{COMPRESSION_RATIO}_e{epoch:04d}.png", c)

    avg_vloss = running_vloss  * 1e4 / TEST_IMAGE_NUM / 10
    print(f'LOSS train {avg_loss:.8f} valid {avg_vloss:.8f}')
    print(f'LOSS valid PSNR {10 * np.log10(1/avg_vloss.item()):.2f} dB. \n')

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'checkpoints/deepjscc_{CHANNEL_TYPE}_snr{SNR}_c{COMPRESSION_RATIO}_e{epoch:03d}.ckpt'
        torch.save(model.state_dict(), model_path)

