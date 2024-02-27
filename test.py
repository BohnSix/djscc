import numpy as np

validation_loss = [0.0257, 0.0093, 0.007, 0.006, 0.0059, 0.0074, 0.0059, 0.0249, 0.0085, 0.0073, 0.0064, 0.0069, 0.0053, 0.0063]

y = np.array([[15.9 , 20.32, 21.55, 22.22, 22.29, 21.31, 22.29],
              [16.04, 20.71, 21.37, 21.94, 21.61, 22.76, 22.01]])

compression_ratios = [0.04, 0.09, 0.17, 0.25, 0.33, 0.42, 0.49]



for i in compression_ratios:
    print(f"""
SNR10dB_C{i}
![验证结果](tmp/validation_snr10_c{i}_e2500.png)
          """)
# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(compression_ratios, y[0], marker=".", label="awgn_snr10")
# plt.plot(compression_ratios, y[1], marker=".", label="awgn_snr20")
# plt.xlabel("Compression Ratios")
# plt.ylabel("PSNR(dB)")
# plt.title("Performance")
# plt.legend()
# plt.savefig("result.png")


# class Channel(nn.Module):
#     def __init__(self, channel_type, channel_snr):
#         super().__init__()
#         self.channel_type = channel_type
#         self.channel_snr = channel_snr
    
#     def awgn(self, x, stddev, lst):
#         cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(lst[0],lst[1],lst[2]*2)).view(np.complex128)
#         cmplx_dist = torch.from_numpy(cmplx_dist)
#         noise = cmplx_dist.cuda() * stddev
#         return x + noise, torch.ones_like(x)
    
#     def fading(self, x, stddev, lst, h=None):
#         inter_shape = x.shape
#         z = x.reshape(inter_shape[0], -1)
#         print(z.shape)
#         z_dim = z.shape[1] // 2
#         z_in = torch.complex(z[:, :z_dim], z[:, z_dim:])
#         print(z_in.shape)
#         # z_norm = torch.sum(torch.real(z_in * torch.mH(z_in)))  # TODO: z norm
#         # print(z_norm.shape)
        
#         if h is None:  # TODO: check this
#             h = torch.randn(z_in.shape, dtype=torch.complex128)  
#         awgn = torch.randn(z_in.shape, dtype=torch.complex128)
        
#         z_out = h * z_in + awgn

#         z_out = torch.concat([torch.real(z_out), torch.imag(z_out)], 0).reshape(inter_shape)

#         return z_out, h
    
#     def forward(self, channal_input, lst):
#         # print("channel_snr: {}".format(self.channel_snr))
#         k = np.prod(lst)
#         snr = 10**(self.channel_snr/10.0)
#         abs_val = torch.abs(channal_input)
#         signl_pwr = torch.sum(torch.square(abs_val)) / k
#         noise_pwr = signl_pwr / snr
#         noise_stddev = torch.sqrt(noise_pwr)


#         if self.channel_type == "awgn":
#             channal_output, h = self.awgn(channal_input, noise_stddev, lst)
#         elif self.channel_type == "fading":
#             channal_output, h = self.fading(channal_input, noise_stddev, lst)

        
#         return channal_output, h
    

# z = torch.zeros([32, 8, 8, 8]).cuda()
# lst = list(z.shape)[1:]
# channel = Channel("fading", 10).cuda()
# outputs, h = channel(z, lst)
# print(outputs.shape)


