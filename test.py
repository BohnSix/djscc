import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
from scipy.special import erfinv
from commpy.modulation import QAMModem

def snr_to_noise_var(snr_db):
    # 将SNR(dB)转换为噪声方差
    return 10.0**(-snr_db / 10.0)

def simulate_ldpc_qam_awgn(codelength, coderate, snr_dbs):
    n = codelength
    k = int(codelength * coderate)
    d_v = 4
    d_c = int(d_v / coderate)

    # 生成LDPC码
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    # 初始化16QAM调制解调器
    qam_modem = QAMModem(16)

    for snr_db in snr_dbs:
        noise_var = snr_to_noise_var(snr_db)

        # 生成随机信息位
        _, k = G.shape
        v = np.random.randint(0, 2, k)

        # LDPC编码
        encoded_bits = encode(G, v, snr_db)

        # 确保encoded_bits的长度是4的倍数，以适应16-QAM调制
        encoded_bits = np.where(encoded_bits > 0, 1, 0)

        # 16QAM调制
        modulated_signal = qam_modem.modulate(encoded_bits)

        # 通过AWGN信道
        noise = np.sqrt(noise_var / 2) * (np.random.randn(*modulated_signal.shape) + 1j * np.random.randn(*modulated_signal.shape))
        received_signal = modulated_signal + noise

        # 16QAM解调
        demodulated_bits = qam_modem.demodulate(received_signal, demod_type='hard')

        # LDPC解码
        decoded_bits = decode(H, demodulated_bits, snr_db, maxiter=10)
        decoded_message = get_message(G, decoded_bits)

        # 计算误比特率（BER）
        ber = np.mean(v != decoded_message)

        print(f"SNR: {snr_db} dB, BER: {ber}")

# 信号长度和码率
codelength = 3072
coderates = [1/2, 1/3, 1/6, 1/12]
snr_dbs = [0, 10, 20]

for coderate in coderates:
    print(f"\nCode rate: {coderate:.4f}")
    simulate_ldpc_qam_awgn(codelength, coderate, snr_dbs)
