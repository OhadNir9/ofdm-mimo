#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib as mpl
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))


import numpy as np
import matplotlib.pyplot as plt

def OFDM_PAPR(const_size=256, n_fft=2048):
    # Parameters
    #const_size = 256  # 4, 16, 64, 256, 1024
    #n_fft = 2048
    N = 10000  # Number of OFDM Symbols
    OS_factor = 4

    OS_fft = n_fft * OS_factor

    OneDimLimit = int(np.sqrt(const_size)) - 1

    # QAM Modulator (no need to normalize for PAPR computation...)
    FreqDomainData = (
        np.random.choice(np.arange(-OneDimLimit, OneDimLimit + 1, 2), (n_fft, N))
        + 1j * np.random.choice(np.arange(-OneDimLimit, OneDimLimit + 1, 2), (n_fft, N))
    )

    # Oversampling
    FreqDomainDataOverSampled = np.zeros((OS_fft, N), dtype=complex)
    FreqDomainDataOverSampled[:n_fft // 2, :] = FreqDomainData[:n_fft // 2, :]
    FreqDomainDataOverSampled[-n_fft // 2:, :] = FreqDomainData[-n_fft // 2:, :]

    # OFDM Modulator
    TimeDomainMat = np.fft.ifft(FreqDomainDataOverSampled, axis=0)

    # Compute the PAPR
    PAPRdB = np.zeros(N)
    for k in range(TimeDomainMat.shape[1]):
        max_power = np.max(np.abs(TimeDomainMat[:, k])**2)
        avg_power = np.mean(np.abs(TimeDomainMat[:, k])**2)
        PAPRdB[k] = 10 * np.log10(max_power / avg_power)

    # Compute CDF and plot
    CDF1, SNRdBvec1 = MyCDF(PAPRdB)
    return CDF1, SNRdBvec1
    
    plt.semilogy(SNRdBvec1, CDF1)
    plt.grid()
    plt.xlabel('PAPR_0 (dB)')
    plt.ylabel('Prob(PAPR > PAPR_0)')
    plt.title(f'PAPR in an OFDM system with {n_fft} SCs and QAM {const_size}')
    plt.show()

def MyCDF(data):
    SNRdBvec = np.arange(1, np.max(data) + 1, 0.01)
    cdfout = np.array([np.sum(data > snr) for snr in SNRdBvec])
    cdfout = cdfout / len(data)
    return cdfout, SNRdBvec

# Call the function
for const_size in [4, 16, 256]:
    CDF1, SNRdBvec1 = OFDM_PAPR(const_size=const_size)
    plt.semilogy(SNRdBvec1, CDF1, label=f'const_size {const_size}')

plt.grid()
plt.xlabel('PAPR_0 (dB)')
plt.ylabel('Prob(PAPR > PAPR_0)')
plt.title(f'PAPR in an OFDM system with 2048 SCs and different QAM constellation sizes')
plt.legend()
plt.show()

for n_fft in [64, 256, 1024, 2048]:
    CDF1, SNRdBvec1 = OFDM_PAPR(n_fft=n_fft)
    plt.semilogy(SNRdBvec1, CDF1, label=f'N_FFT {n_fft}')

plt.grid()
plt.xlabel('PAPR_0 (dB)')
plt.ylabel('Prob(PAPR > PAPR_0)')
plt.title(f'PAPR in an OFDM system with various SCs number and QAM 256')
plt.legend()
plt.show()

