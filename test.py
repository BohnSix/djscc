import torch
import numpy as np
from torch import nn
from glob import glob



class Channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super().__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr
        self.snr = 10**(self.channel_snr/10.0)
    
    def awgn(self, channel_input, stddev):
        cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(2*len(channel_input))).view(np.complex128)
        cmplx_dist = torch.from_numpy(cmplx_dist).cuda()
        noise = cmplx_dist * stddev
        return channel_input + noise, torch.ones_like(channel_input)
    
    def fading(self, x, stddev, h=None):
        z = torch.real(x)
        z_dim = len(z) // 2
        z_in = torch.complex(z[:z_dim], z[z_dim:])
        
        if h is None:
            h = torch.complex(torch.from_numpy(np.random.normal(0, np.sqrt(2)/2, z_in.shape)), 
                              torch.from_numpy(np.random.normal(0, np.sqrt(2)/2, z_in.shape)))
        noise = torch.complex(torch.from_numpy(np.random.normal(0, np.sqrt(2)/2, z_in.shape)), 
                              torch.from_numpy(np.random.normal(0, np.sqrt(2)/2, z_in.shape)))
        h, noise = h.cuda(), noise.cuda()
        z_out = h * z_in + noise * stddev

        z_out = torch.concat([torch.real(z_out), torch.imag(z_out)], 0)

        return z_out, h
    
    def forward(self, channel_input):
        # print("channel_snr: {}".format(self.channel_snr))
       
        signl_pwr = torch.mean(torch.square(torch.abs(channel_input)))
        noise_pwr = signl_pwr / self.snr
        noise_stddev = torch.sqrt(noise_pwr)

        if self.channel_type == "awgn":
            channal_output, H = self.awgn(channel_input, noise_stddev)
        elif self.channel_type == "fading":
            channal_output, H = self.fading(channel_input, noise_stddev)

        return channal_output, H

P = 1
chn_in = np.random.normal(0, 2, [32, 24, 8, 16]).view(np.complex128)
chn_in = torch.from_numpy(chn_in)
print(torch.mean(torch.square(torch.abs(chn_in))))
lst = list(chn_in.shape)
chn_in = chn_in.flatten()
enery = torch.sum(torch.square(torch.abs(chn_in)))
normalization_factor = np.sqrt(len(chn_in)*P) / torch.sqrt(enery)
chn_in = chn_in * normalization_factor
print(torch.mean(torch.square(torch.abs(chn_in))))

channel = Channel("fading", 20).cuda()

chn_output, H = channel(chn_in.cuda())
chn_output = chn_output.reshape(lst)

print(chn_output.shape)