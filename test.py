import numpy as np
import torch
from torch import nn

class Channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super().__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr
    
    def awgn(self, channel_input, stddev):
        cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(2*len(channel_input))).view(np.complex128)
        cmplx_dist = torch.from_numpy(cmplx_dist).cuda()
        noise = cmplx_dist * stddev
        return channel_input + noise, torch.ones_like(channel_input)
    
    def fading(self, x, stddev, lst, h=None):
        inter_shape = x.shape
        z = x.reshape(inter_shape[0], -1)
        print(z.shape)
        z_dim = z.shape[1] // 2
        z_in = torch.complex(z[:, :z_dim], z[:, z_dim:])
        print(z_in.shape)
        # z_norm = torch.sum(torch.real(z_in * torch.mH(z_in)))  # TODO: z norm
        # print(z_norm.shape)
        
        if h is None:  # TODO: check this
            h = torch.randn(z_in.shape, dtype=torch.complex128)  
        awgn = torch.randn(z_in.shape, dtype=torch.complex128)
        
        z_out = h * z_in + awgn

        z_out = torch.concat([torch.real(z_out), torch.imag(z_out)], 0).reshape(inter_shape)

        return z_out, h

    def powerConstraint(self, channel_input, P):
        print("Average power before:", torch.mean(torch.square(torch.abs(channel_input))).item())
        # norm by total power instead of average power
        enery = torch.sum(torch.square(torch.abs(channel_input)))
        normalization_factor = np.sqrt(len(channel_input)*P) / torch.sqrt(enery)
        channel_input = channel_input * normalization_factor

        # the average power of output should be about P
        print("Average power after:", torch.mean(torch.square(torch.abs(channel_input))).item())
        
        return channel_input
    
    def forward(self, channel_input, P=1):
        # print("channel_snr: {}".format(self.channel_snr))
        lst = list(channel_input.shape)
       
        snr = 10**(self.channel_snr/10.0)

        channel_input = channel_input.flatten()
        channel_input = self.powerConstraint(channel_input, P)

        signl_pwr = torch.mean(torch.square(torch.abs(channel_input)))
        noise_pwr = signl_pwr / snr
        noise_stddev = torch.sqrt(noise_pwr)

        if self.channel_type == "awgn":
            channal_output, h = self.awgn(channel_input, noise_stddev)
        elif self.channel_type == "fading":
            channal_output, h = self.fading(channel_input, noise_stddev, lst)

        channal_output = channal_output.reshape(lst)
        
        return channal_output, h
    
k = 400
P = 1

a = np.random.normal(0, np.sqrt(2), [200, 400])
b = np.random.normal(0, np.sqrt(2), [200, 400])
channel_input = torch.complex(torch.from_numpy(a), torch.from_numpy(b)).cuda()
channel = Channel("awgn", 10).cuda()

channel_output = channel(channel_input, 1)

print("Done")