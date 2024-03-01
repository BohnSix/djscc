# Still on construction...

<!-- ## 模型架构应该没问题了，还需要处理一下功率约束的事。
`Chap.III` 提到 `The encoder maps the n-dimensional input image x to a k-length vector of complex-valued channel input samples z`，也就是把一张`[3x32x32]`的图片映射成一个`[kx1]`的复向量并进行功率约束，这里`n=3x32x32=3072`。

`Chap.III` 中给出的功率约束的公式是
 $z = \sqrt{kP}\frac{\tilde{z}}{\sqrt{\tilde{z}^*\tilde{z}}}$，
其中 $\tilde{z}$ 应该是一个`[kx1]`的向量，分母计算得到的模长是一个常数，保证整个分数的功率为`1`，~~并根据系数 $\sqrt{kP}$ 进行功率约束~~。并计算信号的总功率进行约束。

一张图片编码成 `k` 个符号，信号的平均功率为 `P`，总功率为 `kP`。一次传输 `b` 张图片需要编码成`bK`个符号，信号的平均功率为 `P`，总功率为 `bkP`。~~因此归一化因子应该是 $\frac{1}{\sqrt{bkP}}$。~~并计算信号的总功率进行约束。

传输的信号拉直成`[bk, 1]`，信噪比为`snr`，噪声向量的形状也是`[bk, 1]`，总功率为`bkP/snr`，即噪声服从高斯分布$N(0, \sqrt{bkP})$.


##  训练过程应该没有问题，需要重构并封装一下 -->
<!-- 
在瑞利衰落信道模型中，信道矩阵的元素通常不是直接服从正态分布 $N(0, 1)$。相反，瑞利衰落信道模型中的信道矩阵元素的幅度服从瑞利分布，而相位则在 $[0,2\pi)$ 范围内均匀分布。这是因为瑞利衰落信道模型通常用于描述在无视距（Line of Sight, LoS）条件下的多径传播环境，其中多个反射和散射的信号分量相互叠加，导致接收信号的幅度和相位发生变化。

在瑞利衰落模型中，每个信道矩阵元素可以视为多个独立同分布的复数随机变量之和，其中实部和虚部各自独立地遵循均值为$0$、方差为 $\sigma^2/2$ 的正态分布。这里的 $\sigma^2$ 表示信号分量的总功率，通常取决于特定的环境和系统参数。当这些独立的复数随机变量叠加时，由于中心极限定理，复信号的幅度将遵循瑞利分布，而相位将在 [0,2π) 范围内均匀分布。

因此，如果你正在处理瑞利衰落信道的信道矩阵，你应该期望每个元素的实部和虚部分别服从均值为0、方差为 $\sigma^2/2$ 的正态分布，从而使每个元素的幅度服从瑞利分布。方差 $\sigma^2/2$ 的选择取决于信道的具体条件和系统要求，而不一定是1。如果  $\sigma^2=1$ ，那么实部和虚部的方差就是 $1/2$，这种情况下信道模型反映了单位总功率的假设。 -->




# Launch Records

## Introduction

Reimplement [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://arxiv.org/abs/1809.01733) in Pytorch, but **FAILED** to reach the performance mentioned in literature.

Thanks to [irdanish11's implemantation](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission) and [Ahmedest61's implemantation](https://github.com/Ahmedest61/D-JSCC). 

![djscc_performance](resources/djscc_performance.png)



## Technical Solution

Using an `AutoEncoder`to compress image from `[b, 3, H, W]` to feature maps with shape of`[b, c, h, w]`, feed into channels `[AWGN, Slow Fading Channel]` after power constraint and recover.


## Experimental setup

Use `Adam` optimizer，`batch size` set to `8192`, `learning rate` set to `1e-3`, and update to `1e-4` after 1000 `epoch`. Train 2500 `epoch` in total.

Train with `SNR` and `compression rate`, where`SNR`varies in `[0, 10, 20]`，`compression rate` varies in `[0.04, 0.09, 0.17, 0.25, 0.33, 0.42, 0.49]`, namely `channel width` varies in `[2, 4, 8, 12, 16, 20, 24]`.

During performance evaluation transmit each image `10` times in order to mitigate the effect of randomness introduced by the communication channel.


## Model Metric

- Loss Function：`MSE Loss`

- Performance Metric：`PSNR`

- Computational Cost：`7s * 2500epochs / 3600 ~= 5h` on a single `4090Ti`

## Experimental result

**Unable to achieve the performance mentioned in literature**.

See [Visualization](visualization.md) for details.

![Validation Loss](resources/valid_loss.png)

![Model performance](result.png)



<!-- 
## colab environment setup snippets

This is used to install an old version python on colab for tensorflow 1.15.

```
%env PYTHONPATH = # /env/python
!wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
!chmod +x Miniconda3-py38_4.12.0-Linux-x86_64.sh
!./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -f -p /usr/local
!conda update conda -y
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
!conda create -n myenv python=3.6 -y
```
```
%%shell
eval "$(conda shell.bash hook)"
conda activate myenv
pip install tensorflow==1.15 -q
``` -->
