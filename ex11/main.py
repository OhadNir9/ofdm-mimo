#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib as mpl
import traceback
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

def OFDMA_MIMO():
    N_fft = 256
    N = 100  # Num of OFDM Symbols
    CP_length = 16
    NumRx = 4

    # SC indices of STA0 and STA1
    STA0loc = np.arange(0, 50)  # MATLAB [1:50] is Python [0:50)
    STA1loc = np.arange(50, 70)  # MATLAB [51:70] is Python [50:70)

    # Frequency Domain Data
    s0 = np.zeros((N_fft, N), dtype=complex)
    s0[STA0loc, :] = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], (len(STA0loc), N)) / np.sqrt(2))

    s1_0 = np.zeros((N_fft, N), dtype=complex)
    s1_0[STA1loc, :] = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], (len(STA1loc), N)) / np.sqrt(2))
    s1_0[:, 1] = 0  # Second symbol is null

    s1_1 = np.zeros((N_fft, N), dtype=complex)
    s1_1[STA1loc, :] = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], (len(STA1loc), N)) / np.sqrt(2))
    s1_1[:, 0] = 0  # First symbol is null

    # OFDM Modulator
    TimeDomainMat0 = np.fft.ifft(s0, axis=0)
    TimeDomainMat_withCP0 = np.vstack([TimeDomainMat0[-CP_length:, :], TimeDomainMat0])
    TimeDomainSignalLong0 = TimeDomainMat_withCP0.flatten()

    TimeDomainMat1_0 = np.fft.ifft(s1_0, axis=0)
    TimeDomainMat_withCP1_0 = np.vstack([TimeDomainMat1_0[-CP_length:, :], TimeDomainMat1_0])
    TimeDomainSignalLong1_0 = TimeDomainMat_withCP1_0.flatten()

    TimeDomainMat1_1 = np.fft.ifft(s1_1, axis=0)
    TimeDomainMat_withCP1_1 = np.vstack([TimeDomainMat1_1[-CP_length:, :], TimeDomainMat1_1])
    TimeDomainSignalLong1_1 = TimeDomainMat_withCP1_1.flatten()

    # Pass Through MIMO Channel
    RxSignalLong = PassThroughChannel(TimeDomainSignalLong0.reshape(1, -1), NumRx, 10, np.arange(CP_length))
    RxSignalLong += PassThroughChannel(np.vstack([TimeDomainSignalLong1_0, TimeDomainSignalLong1_1]), NumRx, 10, np.arange(CP_length))

    # OFDMA Receiver
    BigPostFFTRx = np.zeros((N_fft, N, NumRx), dtype=complex)

    for k in range(NumRx):
        RxSignalMat = RxSignalLong[:, k].reshape(N_fft + CP_length, N)
        RxSignalMatWithoutCP = RxSignalMat[CP_length:, :]
        PostFFTRx = np.fft.fft(RxSignalMatWithoutCP, axis=0)
        BigPostFFTRx[:, :, k] = PostFFTRx

    # Channel Estimation for STA0
    ChannelEstimate0 = np.zeros((NumRx, len(STA0loc)), dtype=complex)
    for k in range(len(STA0loc)):
        ChannelEstimate0[:, k] = BigPostFFTRx[STA0loc[k], 0, :] / s0[STA0loc[k], 0]

    # Plot Channel Estimates
    #plt.stem(np.abs(ChannelEstimate0[0, :]), label="Channel to Rx0")
    #plt.stem(np.abs(ChannelEstimate0[1, :]), linefmt='r', markerfmt='ro', label="Channel to Rx1")
    #plt.title("Estimated Channels of STA0")
    #plt.xlabel("SC Num")
    #plt.ylabel("abs(Channel)")
    #plt.legend()
    #plt.show()

    # Channel Estimation for STA1
    ChannelEstimate1_0 = np.zeros((NumRx, len(STA1loc)), dtype=complex)
    ChannelEstimate1_1 = np.zeros((NumRx, len(STA1loc)), dtype=complex)
    for k in range(len(STA1loc)):
        ChannelEstimate1_0[:, k] = BigPostFFTRx[STA1loc[k], 0, :] / s1_0[STA1loc[k], 0]
        ChannelEstimate1_1[:, k] = BigPostFFTRx[STA1loc[k], 1, :] / s1_1[STA1loc[k], 1]

    # Plot Channel Estimates
    #plt.stem(np.abs(ChannelEstimate1_0[0, :]), label="Channel to Rx0")
    #plt.stem(np.abs(ChannelEstimate1_0[1, :]), linefmt='r', markerfmt='ro', label="Channel to Rx1")
    #plt.title("Estimated Channels of STA1")
    #plt.xlabel("SC Num")
    #plt.ylabel("abs(Channel)")
    #plt.legend()
    #plt.show()


    # Demodulation for STA0
    Demodulated0 = np.zeros((len(STA0loc), N), dtype=complex)
    for k in range(len(STA0loc)):
        for kk in range(N):
            Demodulated0[k, kk] = (ChannelEstimate0[:, k].conj().T @ BigPostFFTRx[STA0loc[k], kk, :]) / (np.linalg.norm(ChannelEstimate0[:, k])**2)

    Payload0 = Demodulated0[:, 1:]
    #plt.plot(Payload0.flatten().real, Payload0.flatten().imag, '.', label="STA0")
    #plt.show()

    # Demodulation for STA1
    Demodulated1 = np.zeros((len(STA1loc), N, 2), dtype=complex)
    for k in range(len(STA1loc)):
        CurrentMIMOChannel = np.column_stack((ChannelEstimate1_0[:, k], ChannelEstimate1_1[:, k]))
        for kk in range(N):
            CurrentY = BigPostFFTRx[STA1loc[k], kk, :]
            Demodulated1[k, kk, :] = np.linalg.pinv(CurrentMIMOChannel) @ CurrentY

    Payload1_0 = Demodulated1[:, 2:, 0].flatten()
    Payload1_1 = Demodulated1[:, 2:, 1].flatten()

    plt.plot(0.99 * Payload1_0.real, 0.99 * Payload1_0.imag, 'r.', label="STA1 Stream0")
    plt.plot(0.98 * Payload1_1.real, 0.98 * Payload1_1.imag, 'k.', label="STA1 Stream1")
    plt.title("Demodulated QAMs (Noiseless Version)")
    plt.legend()
    plt.show()

def PassThroughChannel(TxSignal, NumRx, NumPaths, DelayRange):
    NumTx = TxSignal.shape[0]
    PathsDelay = np.random.choice(DelayRange, NumPaths)
    PathsPhase = np.exp(1j * 2 * np.pi * np.random.rand(NumPaths))
    DoAs = 2 * np.pi * np.random.rand(NumPaths)
    DoDs = 2 * np.pi * np.random.rand(NumPaths)

    RxSignal = np.zeros((NumRx, TxSignal.shape[1]), dtype=complex)
    for k in range(NumPaths):
        print(f"path {k}")
        for kk in range(NumTx):
            print(f"TX idx: {kk}")
            CurrentChunk = TxSignal[kk, :]
            if PathsDelay[k] != 0:
                delayed_chunk = np.concatenate([np.zeros(PathsDelay[k]), CurrentChunk[:-PathsDelay[k]]])
            else:
                delayed_chunk = CurrentChunk
            a = (np.exp(-1j * np.pi * np.sin(DoDs[k]) * kk) * 
                         PathsPhase[k])
            b = np.exp(-1j * np.pi * np.sin(DoAs[k]) * np.arange(NumRx)[:, None])
            c = delayed_chunk[None, :]
            try:
                RxSignal += a*b*c
            
            except Exception as e:
                traceback.print_exc()
                print(f"Exception {e}")
                print(f"a: {a.shape} | b: {b.shape} | c: {c.shape} | RxSignal: {RxSignal.shape}")
                exit()
            
        print("----------------")
    return RxSignal.T


OFDMA_MIMO()