import numpy as np
import torch
from torch import nn

class Channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super().__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr
    
    def awgn(self, x, stddev):
        b, c, h, w = x.shape
        cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(b, c, h, 2*w)).view(np.complex128)
        cmplx_dist = torch.from_numpy(cmplx_dist).cuda()
        noise = cmplx_dist * stddev
        return x + noise, torch.ones_like(x)
    
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
    
    def forward(self, channal_input, P):
        # print("channel_snr: {}".format(self.channel_snr))
        lst = list(channal_input.shape)
        k = np.prod(lst)
       
        snr = 10**(self.channel_snr/10.0)
        abs_val = torch.abs(channal_input)
        signl_pwr = torch.sum(torch.square(abs_val)) / k
        noise_pwr = signl_pwr / snr
        noise_stddev = torch.sqrt(noise_pwr)

        if self.channel_type == "awgn":
            channal_output, h = self.awgn(channal_input, noise_stddev, lst)
        elif self.channel_type == "fading":
            channal_output, h = self.fading(channal_input, noise_stddev, lst)
        
        return channal_output, h
    

# x = torch.randn([32, 8, 8, 8]).cuda()
# z = torch.complex(x, torch.zeros_like(x))
# lst = list(z.shape)[1:]
# channel = Channel("awgn", 20).cuda()
# outputs, h = channel(z, lst)
# print(outputs.shape)


# a = torch.randint(10, [2, 4]).float()
# b = torch.randint(10, [2, 4]).float()

k = 1
P = 1

a = np.random.normal(0, np.sqrt(1/2), [200, 4000])
b = np.random.normal(0, np.sqrt(1/2), [200, 4000])
c = torch.complex(torch.from_numpy(a), torch.from_numpy(b))
c = c.flatten()

c_H = torch.conj(c).T
norm = torch.sqrt(torch.matmul(c_H, c))
norm = torch.norm(c)
print(norm)
normalization_factor = np.sqrt(k*P) / norm
z_hat = c * normalization_factor.unsqueeze(-1)  # 使用 unsqueeze 来确保维度匹配

print(torch.mean(torch.square(torch.abs(c))))

print("Done")