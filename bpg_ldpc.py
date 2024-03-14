from tqdm.auto import tqdm
import os
import math
import numpy as np

from PIL import Image
import tensorflow as tf
from tensorflow import keras

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import ebnodb2no
from sionna.channel import AWGN, FlatFadingChannel


def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1

    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image


class BPGEncoder():
    def __init__(self, working_directory='./analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory

    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgenc {input_dir} -q {qp} -o {output_dir} -f 444')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def get_qp(self, input_dir, byte_threshold, output_dir='temp.bpg'):
        '''
        iteratively finds quality parameter that maximizes quality given the byte_threshold constraint
        '''
        # rate-match algorithm
        quality_max = 51
        quality_min = 0
        quality = (quality_max - quality_min) // 2

        while True:
            qp = 51 - quality
            bytes = self.run_bpgenc(qp, input_dir, output_dir)
            if quality == 0 or quality == quality_min or quality == quality_max:
                break
            elif bytes > byte_threshold and quality_min != quality - 1:
                quality_max = quality
                quality -= (quality - quality_min) // 2
            elif bytes > byte_threshold and quality_min == quality - 1:
                quality_max = quality
                quality -= 1
            elif bytes < byte_threshold and quality_max > quality:
                quality_min = quality
                quality += (quality_max - quality) // 2
            else:
                break

        return qp

    def encode(self, image_array, max_bytes, header_bytes=22):
        '''
        image_array: uint8 numpy array with shape (b, h, w, c)
        max_bytes: int, maximum bytes of the encoded image file (exlcuding header bytes)
        header_bytes: the size of BPG header bytes (to be excluded in image file size calculation)
        '''

        input_dir = f'{self.working_directory}/temp_enc.png'
        output_dir = f'{self.working_directory}/temp_enc.bpg'

        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)

        if self.run_bpgenc(qp, input_dir, output_dir) < 0:
            raise RuntimeError("BPG encoding failed")

        # read binary and convert it to numpy binary array with float dtype
        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)


class LDPCTransmitter():
    '''
    Transmits given bits (float array of '0' and '1') with LDPC.
    '''

    def __init__(self, k, n, m, esno_db, channel='AWGN'):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM)
        esno_db: channel SNR
        channel: 'AWGN' or 'Rayleigh'
        '''
        self.k = k
        self.n = n
        self.num_bits_per_symbol = round(math.log2(m))

        constellation_type = 'qam' if m != 2 else 'pam'
        self.constellation = Constellation(
            constellation_type, num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper('app', constellation=self.constellation)
        self.channel = AWGN() if channel == 'AWGN' else FlatFadingChannel
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
        self.esno_db = esno_db

    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(
            source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        no = ebnodb2no(self.esno_db, num_bits_per_symbol=1, coderate=1)
        c = self.encoder(u)
        x = self.mapper(c)
        y = self.channel([x, no])
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]


class BPGDecoder():
    def __init__(self, working_directory='./analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory

    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'bpgdec {input_dir} -o {output_dir}')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def decode(self, bit_array, image_shape):
        '''
        returns decoded result of given bit_array.
        if bit_array is not decodable, then returns the mean CIFAR-10 pixel values.

        byte_array: float array of '0' and '1'
        image_shape: used to generate image with mean pixel values if the given byte_array is not decodable
        '''
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        with open(input_dir, "wb") as binary_file:
            binary_file.write(byte_array.tobytes())

        cifar_mean = np.array(
            [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255
        cifar_mean = np.reshape(
            cifar_mean, [1] * (len(image_shape) - 1) + [3]).astype(np.uint8)

        if self.run_bpgdec(input_dir, output_dir) < 0:
            # print('warning: Decode failed. Returning mean pixel value')
            return 0 * np.ones(image_shape) + cifar_mean
        else:
            x = np.array(Image.open(output_dir).convert('RGB'))
            if x.shape != image_shape:
                return 0 * np.ones(image_shape) + cifar_mean
            return x


(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# BPG + LDPC
bpgencoder = BPGEncoder()
bpgdecoder = BPGDecoder()

bw_ratio = [1/12, 1/6, 1/4, 1/3, 1/2]
snrs = [0, 10]
mcs = [(k, n, m) for k, n in [(3072, 6144), (3072, 4608), (1536, 4608)]
       for m in (2, 4, 16, 64)]
batchsize = 256
'''
(3072, 6144), (3072, 4608), (1536, 4608)
BPSK, 4-QAM, 16-QAM, 64-QAM
'''


for esno_db in snrs:
    for bw in bw_ratio:
        for k, n, m in mcs:
            i = 0
            psnr = 0
            ssim = 0
            total_images = 0
            ldpctransmitter = LDPCTransmitter(k, n, m, esno_db, 'AWGN')
            # for image, _ in tqdm(trainloader):
            # for image, _ in [(train_images, train_labels)]:
            for start in range(len(train_images)//batchsize):
                end = (start+1)*batchsize
                start = start*batchsize
                image = train_images[start:end]
                b, _, _, _ = image.shape
                image = tf.cast(imBatchtoImage(image), tf.uint8)
                max_bytes = b * 32 * 32 * 3 * bw * math.log2(m) * k / n / 8
                src_bits = bpgencoder.encode(image.numpy(), max_bytes)
                rcv_bits = ldpctransmitter.send(src_bits)

                decoded_image = bpgdecoder.decode(
                    rcv_bits.numpy(), image.shape)
                total_images += b
                psnr = (total_images - b) / (total_images) * psnr + float(b *
                                                                          tf.image.psnr(decoded_image, image, max_val=255)) / (total_images)
                ssim = (total_images - b) / (total_images) * ssim + float(b * tf.image.ssim(tf.cast(
                    decoded_image, dtype=tf.float32), tf.cast(image, dtype=tf.float32), max_val=255)) / (total_images)

            print(
                f'[res] SNR={esno_db}, bw={bw:.6f}, k={k}, n={n}, m={m}, PSNR={psnr:.2f}, SSIM={ssim:.2f}')




sss = """
SNR=00, bw=0.083333, k=3072, n=6144, m=02, PSNR=19.61, SSIM=0.59
SNR=00, bw=0.083333, k=3072, n=6144, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=3072, n=6144, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=3072, n=4608, m=02, PSNR=7.79, SSIM=0.10
SNR=00, bw=0.083333, k=3072, n=4608, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=3072, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=1536, n=4608, m=02, PSNR=19.61, SSIM=0.59
SNR=00, bw=0.083333, k=1536, n=4608, m=04, PSNR=20.09, SSIM=0.62
SNR=00, bw=0.083333, k=1536, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.083333, k=1536, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=6144, m=02, PSNR=21.56, SSIM=0.71
SNR=00, bw=0.166667, k=3072, n=6144, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=6144, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=4608, m=02, PSNR=7.64, SSIM=0.09
SNR=00, bw=0.166667, k=3072, n=4608, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=1536, n=4608, m=02, PSNR=20.09, SSIM=0.62
SNR=00, bw=0.166667, k=1536, n=4608, m=04, PSNR=22.40, SSIM=0.75
SNR=00, bw=0.166667, k=1536, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.166667, k=1536, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=6144, m=02, PSNR=22.90, SSIM=0.77
SNR=00, bw=0.250000, k=3072, n=6144, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=6144, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=4608, m=02, PSNR=7.60, SSIM=0.08
SNR=00, bw=0.250000, k=3072, n=4608, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=1536, n=4608, m=02, PSNR=21.56, SSIM=0.71
SNR=00, bw=0.250000, k=1536, n=4608, m=04, PSNR=24.17, SSIM=0.82
SNR=00, bw=0.250000, k=1536, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.250000, k=1536, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=6144, m=02, PSNR=24.17, SSIM=0.82
SNR=00, bw=0.333333, k=3072, n=6144, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=6144, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=4608, m=02, PSNR=7.54, SSIM=0.08
SNR=00, bw=0.333333, k=3072, n=4608, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=1536, n=4608, m=02, PSNR=22.40, SSIM=0.75
SNR=00, bw=0.333333, k=1536, n=4608, m=04, PSNR=25.49, SSIM=0.86
SNR=00, bw=0.333333, k=1536, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.333333, k=1536, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=6144, m=02, PSNR=26.19, SSIM=0.88
SNR=00, bw=0.500000, k=3072, n=6144, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=6144, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=4608, m=02, PSNR=7.50, SSIM=0.07
SNR=00, bw=0.500000, k=3072, n=4608, m=04, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=1536, n=4608, m=02, PSNR=24.17, SSIM=0.82
SNR=00, bw=0.500000, k=1536, n=4608, m=04, PSNR=27.94, SSIM=0.91
SNR=00, bw=0.500000, k=1536, n=4608, m=16, PSNR=6.57, SSIM=0.12
SNR=00, bw=0.500000, k=1536, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.083333, k=3072, n=6144, m=02, PSNR=19.61, SSIM=0.59
SNR=10, bw=0.083333, k=3072, n=6144, m=04, PSNR=21.56, SSIM=0.71
SNR=10, bw=0.083333, k=3072, n=6144, m=16, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.083333, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.083333, k=3072, n=4608, m=02, PSNR=20.09, SSIM=0.62
SNR=10, bw=0.083333, k=3072, n=4608, m=04, PSNR=22.40, SSIM=0.75
SNR=10, bw=0.083333, k=3072, n=4608, m=16, PSNR=25.12, SSIM=0.85
SNR=10, bw=0.083333, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.083333, k=1536, n=4608, m=02, PSNR=19.61, SSIM=0.59
SNR=10, bw=0.083333, k=1536, n=4608, m=04, PSNR=20.09, SSIM=0.62
SNR=10, bw=0.083333, k=1536, n=4608, m=16, PSNR=22.40, SSIM=0.75
SNR=10, bw=0.083333, k=1536, n=4608, m=64, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.166667, k=3072, n=6144, m=02, PSNR=21.56, SSIM=0.71
SNR=10, bw=0.166667, k=3072, n=6144, m=04, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.166667, k=3072, n=6144, m=16, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.166667, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.166667, k=3072, n=4608, m=02, PSNR=22.40, SSIM=0.75
SNR=10, bw=0.166667, k=3072, n=4608, m=04, PSNR=25.49, SSIM=0.86
SNR=10, bw=0.166667, k=3072, n=4608, m=16, PSNR=29.75, SSIM=0.94
SNR=10, bw=0.166667, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.166667, k=1536, n=4608, m=02, PSNR=20.09, SSIM=0.62
SNR=10, bw=0.166667, k=1536, n=4608, m=04, PSNR=22.40, SSIM=0.75
SNR=10, bw=0.166667, k=1536, n=4608, m=16, PSNR=25.49, SSIM=0.86
SNR=10, bw=0.166667, k=1536, n=4608, m=64, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.250000, k=3072, n=6144, m=02, PSNR=22.90, SSIM=0.77
SNR=10, bw=0.250000, k=3072, n=6144, m=04, PSNR=26.19, SSIM=0.88
SNR=10, bw=0.250000, k=3072, n=6144, m=16, PSNR=30.55, SSIM=0.95
SNR=10, bw=0.250000, k=3072, n=6144, m=64, PSNR=6.59, SSIM=0.12
SNR=10, bw=0.250000, k=3072, n=4608, m=02, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.250000, k=3072, n=4608, m=04, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.250000, k=3072, n=4608, m=16, PSNR=32.85, SSIM=0.97
SNR=10, bw=0.250000, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.250000, k=1536, n=4608, m=02, PSNR=21.56, SSIM=0.71
SNR=10, bw=0.250000, k=1536, n=4608, m=04, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.250000, k=1536, n=4608, m=16, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.250000, k=1536, n=4608, m=64, PSNR=30.55, SSIM=0.95
SNR=10, bw=0.333333, k=3072, n=6144, m=02, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.333333, k=3072, n=6144, m=04, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.333333, k=3072, n=6144, m=16, PSNR=32.85, SSIM=0.97
SNR=10, bw=0.333333, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.333333, k=3072, n=4608, m=02, PSNR=25.49, SSIM=0.86
SNR=10, bw=0.333333, k=3072, n=4608, m=04, PSNR=29.75, SSIM=0.94
SNR=10, bw=0.333333, k=3072, n=4608, m=16, PSNR=35.25, SSIM=0.98
SNR=10, bw=0.333333, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.333333, k=1536, n=4608, m=02, PSNR=22.40, SSIM=0.75
SNR=10, bw=0.333333, k=1536, n=4608, m=04, PSNR=25.49, SSIM=0.86
SNR=10, bw=0.333333, k=1536, n=4608, m=16, PSNR=29.75, SSIM=0.94
SNR=10, bw=0.333333, k=1536, n=4608, m=64, PSNR=32.85, SSIM=0.97
SNR=10, bw=0.500000, k=3072, n=6144, m=02, PSNR=26.19, SSIM=0.88
SNR=10, bw=0.500000, k=3072, n=6144, m=04, PSNR=30.55, SSIM=0.95
SNR=10, bw=0.500000, k=3072, n=6144, m=16, PSNR=36.48, SSIM=0.98
SNR=10, bw=0.500000, k=3072, n=6144, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.500000, k=3072, n=4608, m=02, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.500000, k=3072, n=4608, m=04, PSNR=32.85, SSIM=0.97
SNR=10, bw=0.500000, k=3072, n=4608, m=16, PSNR=38.58, SSIM=0.97
SNR=10, bw=0.500000, k=3072, n=4608, m=64, PSNR=6.57, SSIM=0.12
SNR=10, bw=0.500000, k=1536, n=4608, m=02, PSNR=24.17, SSIM=0.82
SNR=10, bw=0.500000, k=1536, n=4608, m=04, PSNR=27.94, SSIM=0.91
SNR=10, bw=0.500000, k=1536, n=4608, m=16, PSNR=32.85, SSIM=0.97
SNR=10, bw=0.500000, k=1536, n=4608, m=64, PSNR=36.48, SSIM=0.98
"""