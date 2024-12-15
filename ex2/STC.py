#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
from utils import diversity_order_from_ser_snr_curve


class STC:
    def __init__(self, N_Rx, SNR_dB, num_of_symbols=100000):
        self.N_Tx = 2  # STC supports only 2 TX antennas.
        self.N_Rx = N_Rx
        self.SNRs_dB = SNR_dB
        self.SNRs = 10**(self.SNRs_dB / 10)
        self.N = num_of_symbols
        self.QPSK_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    def exec(self, plot=False):
        self.SER = np.zeros_like(self.SNRs_dB, dtype=float)
        
        # Loop over SNR points
        for k in range(len(self.SNRs_dB)):
            print(f"{self.SNRs_dB[k]} SNR")
            #s_hat = np.zeros(self.N, dtype=complex)
            s_hat = np.zeros((2, self.N), dtype=complex)  # Detected symbols
            rho = 1 / np.sqrt(self.SNRs[k])

            # Signal Generation
            #s = np.random.choice(self.QPSK_vec, self.N)
            s = np.random.choice(self.QPSK_vec, size=(2,self.N))
            h = (np.random.randn(2, self.N) + 1j * np.random.randn(2, self.N)) / np.sqrt(2)  # Rayleigh channel
            n = (np.random.randn(2, self.N) + 1j * np.random.randn(2, self.N)) / np.sqrt(2)

            for kk in range(self.N):
                # STC channel matrix
                H_stc = (1 / np.sqrt(2)) * np.array([
                    [h[0, kk], h[1, kk]],
                    [np.conj(h[1, kk]), -np.conj(h[0, kk])]
                ])
        
                # Received signal
                y = H_stc @ s[:, kk] + rho * n[:, kk]
        
                # Signal detection
                s_hat[0, kk] = np.conj(H_stc[:, 0]) @ y / np.linalg.norm(H_stc[:, 0])**2
                s_hat[1, kk] = np.conj(H_stc[:, 1]) @ y / np.linalg.norm(H_stc[:, 1])**2

            # Quantization
            s_tilde = np.sign(s_hat.real) / np.sqrt(2) + 1j * np.sign(s_hat.imag) / np.sqrt(2)

            # Error computation
            NumOfErrors = np.sum(np.abs(s_tilde - s) > np.finfo(float).eps)
            self.SER[k] = NumOfErrors / self.N

        if plot:
            # Plot results
            plt.semilogy(self.SNRs_dB, self.SER, marker='o')
            plt.grid()
            plt.xlabel('SNR (dB)')
            plt.ylabel('SER')
            plt.title(f'STC TX: {self.N_Tx} RX: {self.N_Rx} Rayleigh\n \
            Estimated Diversity Order: {diversity_order_from_ser_snr_curve(sim.SER, sim.SNRs_dB):.2f}', loc="center")
            plt.show()
        
        return self.SNRs_dB, self.SER

if __name__ == "__main__":
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Configure simulation parameters: N_Tx, N_Rx, min_snr, max_snr."
    )

    # Add arguments with help messages and default values
    parser.add_argument(
        "--N_Rx", 
        type=int, 
        default=2, 
        help="Number of receive antennas (default: 2)"
    )
    parser.add_argument(
        "--min_snr", 
        type=float, 
        default=0.0, 
        help="Minimum SNR value in dB (default: 0.0)"
    )
    parser.add_argument(
        "--max_snr", 
        type=float, 
        default=9.0, 
        help="Maximum SNR value in dB (default: 9.0)"
    )
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)

    # Parse the arguments
    args = parser.parse_args()

    # Print out the parsed arguments
    print(f"N_Rx: {args.N_Rx}")
    print(f"min_snr: {args.min_snr}")
    print(f"max_snr: {args.max_snr}")
    print(f"plot={args.plot}")
    
    sim = STC(N_Rx=args.N_Rx,SNR_dB=np.arange(args.min_snr,args.max_snr,1))
    sim.exec(plot=args.plot)
    #print(f"diver: {diversity_order_from_ser_snr_curve(sim.SER, sim.SNRs_dB)}")