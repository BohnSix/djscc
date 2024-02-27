# Launch Records

## Introduction

Reimplement a `JSCC` solution for weirless image transmission, but **FAILED** to reach the performance mentioned in literature.

[Deep Joint Source-Channel Coding for Wireless Image Transmission](https://arxiv.org/abs/1809.01733).

![djscc_performance](resources/djscc_performance.png)


Reference： [irdanish11 implemantation](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission) and [Ahmedest61 implemantation](https://github.com/Ahmedest61/D-JSCC)

## Technical Solution

Using an `AutoEncoder`to compress image from `[b, 3, H, W]` to feature maps with the shape of`[b, c, h, w]`, feed into channels `[AWGN, Slow Fading Channel]` after power constraint and recover. 


## Experimental setup

Use `Adam` optimizer，`batch size` set to `8192`,
`learning rate` set to `1e-3`, and update to `1e-4` after 64 `epoch`. Train 2500 `epoch`.

Train with `SNR` and `compression rate`, where`SNR`varies in `[0, 10, 20]`，`compression rate` varies in `[0.04, 0.09, 0.17, 0.25, 0.33, 0.42, 0.49]`, namely `channel width` varies in `[2, 4, 8, 12, 16, 20, 24]`.

During performance evaluation transmit each image 10 times in order to mitigate the effect of randomness introduced by the communication channel.



## Model Metric

- Loss Function：`MSE Loss`

- Performance Metric：`PSNR`

- Computational Cost：`7s * 2500epochs / 3600 ~= 5h` on a single `4090Ti`

## Experimental result

**Unable to achieve the performance mentioned in literature**. 

See [Visualization](visualization.md) for details.

![Validation Loss](resources/valid_loss.png)

![Model performance](result.png)




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
```
