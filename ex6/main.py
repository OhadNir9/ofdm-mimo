#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib as mpl
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))


class PrecodedSpatialMultiplexing:
    def __init__(self, n_rx_tx=2, num_timeslots=50000):
        self.n_rx_tx = n_rx_tx
        self.num_timeslots = num_timeslots
        self.qpsk_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    def _generate_combination_matrix(self):
        """
        Generates a matrix of N_RX_TX rows and 4^N_RX_TX columns.
        Each column is a possible combination of transmitted QPSK symbols.
        """

        # Generate all combinations of the numbers 0-3 for M rows
        combinations_indices = list(product(range(4), repeat=self.n_rx_tx))
        combination_qpsks = self.qpsk_vec[combinations_indices].T
        return combination_qpsks
    
    def generate_transmitted_precoded_data(self, snr_db):
        """
        Generates a matrix with N_RX_TX rows and N_SYMBOLS columns.
        Each column is a 'time slot', and each row within it is the SVD precoded symbol, after channel passage (additive noise..)
        """
        
        snr = 10 ** (snr_db / 10)
        rho = 1 / np.sqrt(snr)
        N = (np.random.randn(self.n_rx_tx, self.num_timeslots) + 1j * np.random.randn(self.n_rx_tx, self.num_timeslots)) / np.sqrt(2)
        S = self.qpsk_vec[np.random.randint(0, 4, (self.n_rx_tx, self.num_timeslots))]
        # Channel matrix
        H = (np.random.randn(self.n_rx_tx, self.n_rx_tx) + 1j * np.random.randn(self.n_rx_tx, self.n_rx_tx)) / np.sqrt(2)
        U, D, Vh = np.linalg.svd(H)
        V = Vh.T.conj()
        H_tilde = np.dot(H, V)  # SVD precoding
        Y = np.dot(H_tilde, S) / np.sqrt(2) + rho * N
        return S, Y, H_tilde
    
    def decode_zf(self, Y, H_tilde):
        S_hat_ZF = np.dot(np.linalg.pinv(H_tilde / np.sqrt(2)), Y)
        S_tilde_ZF = np.sign(np.real(S_hat_ZF)) / np.sqrt(2) + 1j * np.sign(np.imag(S_hat_ZF)) / np.sqrt(2)
        return S_tilde_ZF

        
    def decode_ml(self, Y, H_tilde, qpsk_combs):
        S_tilde_ML = np.zeros((self.n_rx_tx, self.num_timeslots), dtype=complex)
        for timeslot in range(Y.shape[1]):
            diff_vec = Y[:, timeslot][:, np.newaxis] - np.dot(H_tilde / np.sqrt(2), qpsk_combs)
            cost_vec = np.sum(np.abs(diff_vec) ** 2, axis=0)
            min_loc = np.argmin(cost_vec)
            S_tilde_ML[:, timeslot] = qpsk_combs[:, min_loc]
        return S_tilde_ML

    def exec(self, snr_min_db=10, snr_max_db=15, plot=True):
        qpsk_combs = self._generate_combination_matrix()
        snrs_db = list(range(snr_min_db, snr_max_db))
        SER_ZF = np.zeros((self.n_rx_tx, len(snrs_db)), dtype=float)
        SER_ML = np.zeros((self.n_rx_tx, len(snrs_db)), dtype=float)
        for idx, snr_db in enumerate(snrs_db):
            print(f"SNR: {snr_db} dB")
            S, Y, H_tilde = self.generate_transmitted_precoded_data(snr_db=snr_db)
            S_tilde_zf = self.decode_zf(Y, H_tilde)
            S_tilde_ml = self.decode_ml(Y, H_tilde, qpsk_combs)

            num_of_errs_zf = np.zeros(self.n_rx_tx)
            num_of_errs_ml = np.zeros(self.n_rx_tx)
            for stream in range(self.n_rx_tx):
                num_of_errs_zf[stream] = np.sum(np.abs(S_tilde_zf[stream, :] - S[stream, :]) > np.finfo(float).eps)            
                num_of_errs_ml[stream] = np.sum(np.abs(S_tilde_ml[stream, :] - S[stream, :]) > np.finfo(float).eps)

            SER_ZF[:, idx] = num_of_errs_zf / self.num_timeslots
            SER_ML[:, idx] = num_of_errs_ml / self.num_timeslots

            #ser_zf = num_of_errs_zf / self.num_timeslots
            #ser_ml = num_of_errs_ml / self.num_timeslots

            #print("------------------")
            #print(f"SNR={snr_db}dB")
            #print(SER_ZF[:, idx])
            #print(SER_ML[:, idx])

        if plot:
            for stream in range(self.n_rx_tx):
                plt.semilogy(snrs_db, SER_ZF[stream, :], label=f'ZF stream {stream}')
                #plt.scatter(snrs_db, SER_ZF[stream, :], label=f'ZF stream {stream}')
                #plt.scatter(snrs_db, SER_ML[stream, :], label=f'ML stream {stream}')
                plt.semilogy(snrs_db, SER_ML[stream, :], label=f'ML stream {stream}')

        
            plt.grid()
            plt.xlabel('SNR dB')
            plt.ylabel('SER')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Precoded SM {self.n_rx_tx}X{self.n_rx_tx} Rayleigh')
            plt.show()

if __name__ == "__main__":
    psm = PrecodedSpatialMultiplexing(num_timeslots=1000000)
    psm.exec(snr_min_db=0, snr_max_db=40)

