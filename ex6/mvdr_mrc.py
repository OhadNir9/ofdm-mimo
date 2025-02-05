#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib as mpl
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

class MVDR_MRC:
    def __init__(self):
        self.n_rx = 4
        self.sir_db = 5
        #self.sir = 10 ** (self.sir_db / 10)
        self.snrs_db = np.arange(0, 21, 2)
        self.snrs = 10 ** (self.snrs_db / 10)
        self.qpsk_vec = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        self.num_timeslots = 100000

        self.interfere_gain = 10 ** (-self.sir_db / 20)

    def _generate_transmitted_symbols(self, snr_db):
        # Signal Generation
        y = np.zeros((self.n_rx, self.num_timeslots), dtype=complex)
        s = np.random.choice(self.qpsk_vec, size=self.num_timeslots)
        h = (np.random.randn(self.n_rx, self.num_timeslots) + 1j * np.random.randn(self.n_rx, self.num_timeslots)) / np.sqrt(2)
        g = (np.random.randn(self.n_rx, self.num_timeslots) + 1j * np.random.randn(self.n_rx, self.num_timeslots)) / np.sqrt(2)
        r = (np.random.randn(1, self.num_timeslots) + 1j * np.random.randn(1, self.num_timeslots)) / np.sqrt(2)
        n = (np.random.randn(self.n_rx, self.num_timeslots) + 1j * np.random.randn(self.n_rx, self.num_timeslots)) / np.sqrt(2)    
        snr = 10 ** (snr_db / 10)
        rho = 1 / np.sqrt(snr)
        for timeslot in range(self.num_timeslots):
            y[:, timeslot] = h[:, timeslot] * s[timeslot] + self.interfere_gain * g[:, timeslot] * r[0, timeslot] + rho * n[:, timeslot]
        return y, s, h, g, rho
    
    def mrc_detector(self, h, y):
        s_hat_mrc = np.zeros(self.num_timeslots, dtype=complex)
        for timeslot in range(self.num_timeslots):
            s_hat_mrc[timeslot] = np.dot(h[:, timeslot].conj().T, y[:, timeslot]) / np.linalg.norm(h[:, timeslot]) ** 2
        s_tilde_mrc = np.sign(np.real(s_hat_mrc)) / np.sqrt(2) + 1j* np.sign(np.imag(s_hat_mrc)) / np.sqrt(2)
        
        return s_tilde_mrc
    
    def mvdr_detector(self, h, y, g, rho):
        #c = np.zeros((self.n_rx, self.n_rx), dtype=complex)
        s_hat_mvdr = np.zeros(self.num_timeslots, dtype=complex)
        for timeslot in range(self.num_timeslots):
            c = self.interfere_gain ** 2 * np.outer(g[:, timeslot], g[:, timeslot].conj()) + rho ** 2 * np.eye(self.n_rx)
            inv_c = np.linalg.inv(c)
            s_hat_mvdr[timeslot] = np.dot(h[:, timeslot].conj().T, np.dot(inv_c, y[:, timeslot])) / np.dot(h[:, timeslot].conj().T, np.dot(inv_c, h[:, timeslot]))
        s_tilde_mvdr = np.sign(np.real(s_hat_mvdr)) / np.sqrt(2) + 1j* np.sign(np.imag(s_hat_mvdr)) / np.sqrt(2)
        
        return s_tilde_mvdr
    
    def exec(self, snr_db):
        y, s, h, g, rho = sim._generate_transmitted_symbols(snr_db)
        s_tilde_mrc = sim.mrc_detector(h, y)
        s_tilde_mvdr = sim.mvdr_detector(h, y, g, rho)
        num_of_errors_mrc = np.sum(np.abs(s_tilde_mrc - s) > np.finfo(float).eps)
        num_of_errors_mvdr = np.sum(np.abs(s_tilde_mvdr - s) > np.finfo(float).eps)
        ser_mrc = num_of_errors_mrc / sim.num_timeslots
        ser_mvdr = num_of_errors_mvdr / sim.num_timeslots

        return ser_mrc, ser_mvdr
    
"""
# Parameters
N_Rx = 4  # Number of Rx antennas
SIRdB = 5
QPSK_vec = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
SNRdB = np.arange(0, 21, 2)
SNR = 10 ** (SNRdB / 10)

SER_MRC = np.zeros(len(SNRdB))
SER_MVDR = np.zeros(len(SNRdB))
N = 100000

InterfereGain = 10 ** (-SIRdB / 20)

# Main Loop
for k in range(len(SNRdB)):
    s_hat_MRC = np.zeros(N, dtype=complex)
    s_hat_MVDR = np.zeros(N, dtype=complex)
    rho = 1 / np.sqrt(SNR[k])
    
    # Signal Generation
    s = np.random.choice(QPSK_vec, size=N)
    h = (np.random.randn(N_Rx, N) + 1j * np.random.randn(N_Rx, N)) / np.sqrt(2)
    g = (np.random.randn(N_Rx, N) + 1j * np.random.randn(N_Rx, N)) / np.sqrt(2)
    r = (np.random.randn(1, N) + 1j * np.random.randn(1, N)) / np.sqrt(2)
    n = (np.random.randn(N_Rx, N) + 1j * np.random.randn(N_Rx, N)) / np.sqrt(2)
    
    for kk in range(N):
        y = h[:, kk] * s[kk] + InterfereGain * g[:, kk] * r[0, kk] + rho * n[:, kk]
        
        # MRC Signal Detection
        s_hat_MRC[kk] = np.dot(h[:, kk].conj().T, y) / np.linalg.norm(h[:, kk]) ** 2
        
        # MVDR Signal Detection
        C = InterfereGain ** 2 * np.outer(g[:, kk], g[:, kk].conj()) + rho ** 2 * np.eye(N_Rx)
        InvC = np.linalg.inv(C)
        s_hat_MVDR[kk] = np.dot(h[:, kk].conj().T, np.dot(InvC, y)) / np.dot(h[:, kk].conj().T, np.dot(InvC, h[:, kk]))
    
    # Signal Decoding
    s_tilde_MRC = np.sign(np.real(s_hat_MRC)) / np.sqrt(2) + 1j * np.sign(np.imag(s_hat_MRC)) / np.sqrt(2)
    s_tilde_MVDR = np.sign(np.real(s_hat_MVDR)) / np.sqrt(2) + 1j * np.sign(np.imag(s_hat_MVDR)) / np.sqrt(2)
    NumOfErrors_MRC = np.sum(np.abs(s_tilde_MRC - s) > np.finfo(float).eps)
    NumOfErrors_MVDR = np.sum(np.abs(s_tilde_MVDR - s) > np.finfo(float).eps)
    SER_MRC[k] = NumOfErrors_MRC / N
    SER_MVDR[k] = NumOfErrors_MVDR / N

# Plotting Results
plt.semilogy(SNRdB, SER_MRC, label="MRC")
plt.semilogy(SNRdB, SER_MVDR, 'r', label="MVDR")
plt.grid()
plt.xlabel("SNR(dB)")
plt.ylabel("SER")
plt.legend()
plt.title(f"MRC and MVDR Rayleigh {N_Rx} Antennas at SINR={SIRdB}dB")
plt.show()
"""
snrs_db = np.arange(0, 21, 2)
SER_MRC = np.zeros(len(snrs_db), dtype=float)
SER_MVDR = np.zeros(len(snrs_db), dtype=float)
sim = MVDR_MRC()
for idx, snr_db in enumerate(snrs_db):
    print(f"SNR: {snr_db} dB")
    SER_MRC[idx], SER_MVDR[idx] = sim.exec(snr_db)

plt.semilogy(snrs_db, SER_MRC, label="MRC")
plt.semilogy(snrs_db, SER_MVDR, 'r', label="MVDR")
plt.grid()
plt.xlabel("SNR(dB)")
plt.ylabel("SER")
plt.legend()
plt.title(f"MRC and MVDR Rayleigh {sim.n_rx} Antennas at SINR={sim.sir_db}dB")
plt.show()