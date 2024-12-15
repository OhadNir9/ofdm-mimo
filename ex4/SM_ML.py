#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import diversity_order_from_ser_snr_curve


class SM_ML:
    # Spatial Multiplexing - Maximum Likelihood Detector
    # Currently supports only 2x2 - TODO configurable N_Tx and N_Rx
    # Diversity Order N, Array Gain N/M
    def __init__(self, SNR_dB, num_of_symbols=50000):
        self.SNRs_dB = SNR_dB
        self.SNRs = 10**(self.SNRs_dB / 10)
        self.N = num_of_symbols
        self.QPSK_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    def exec(self, plot=False):
        self.SER = np.zeros_like(self.SNRs_dB, dtype=float)
        
        # Create a matrix of 16 combinations for ML
        CombinationLoc = np.zeros((2, 16), dtype=int)
        CombinationLoc[0, :] = np.floor(np.arange(16) / 4).astype(int)
        CombinationLoc[1, :] = np.arange(16) % 4

        # CombinationMat is a 2X16 matrix, where each column is a pair of QPSK symbols.
        # Iterating over all of the combination (total 16 pair combinations - 4x4)
        CombinationMat = self.QPSK_vec[CombinationLoc]


        # Loop over SNR points
        for k in range(len(self.SNRs_dB)):
            print(f"{self.SNRs_dB[k]} SNR")
            s_tilde_ml = np.zeros((2,self.N), dtype=complex)
            #s_hat = np.zeros(self.N, dtype=complex)
            rho = 1 / np.sqrt(self.SNRs[k])

            # Signal Generation
            s = self.QPSK_vec[np.random.randint(0, 4, (2,self.N))]
            n = (np.random.randn(2, self.N) + 1j * np.random.randn(2, self.N)) / np.sqrt(2)

            for kk in range(self.N):
                # Channel generation
                H = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)
                
                # Received signal
                y = H @ (1 / np.sqrt(2) * s[:, kk]) + rho * n[:, kk]
                
                # Signal Detection - Maximum Likelihood
                # DiffVec stores the diff between y to every possible QPSK pair combination
                DiffVec = np.tile(y[:, np.newaxis], (1, 16)) - (1 / np.sqrt(2)) * H @ CombinationMat
                # CostVec is simply the square norm of every column in DiffVec
                CostVec = np.sum(np.abs(DiffVec)**2, axis=0)

                MinLoc = np.argmin(CostVec)
                
                # Choosing the QPSK pair with the lowest cost
                s_tilde_ml[:, kk] = CombinationMat[:, MinLoc]

            # Error computation
            NumOfErrors = np.sum(np.abs(s_tilde_ml - s) > np.finfo(float).eps)
            self.SER[k] = NumOfErrors / (self.N * 2)
            

        if plot:
            # Plot results
            plt.semilogy(self.SNRs_dB, self.SER, marker='o')
            plt.grid()
            plt.xlabel('SNR (dB)')
            plt.ylabel('SER')
            plt.title(f'SM 2x2 Rayleigh - Maximum Likelihood\n \
            Estimated Diversity Order: {diversity_order_from_ser_snr_curve(sim.SER, sim.SNRs_dB):.2f}', loc="center")
            plt.show()
        
        return self.SNRs_dB, self.SER

if __name__ == "__main__":
    import argparse
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Configure simulation parameters: min_snr, max_snr."
    )

    # Add arguments with help messages and default values
    parser.add_argument(
        "--min_snr", 
        type=float, 
        default=0.0, 
        help="Minimum SNR value in dB (default: 0.0)"
    )
    parser.add_argument(
        "--max_snr", 
        type=float, 
        default=32, 
        help="Maximum SNR value in dB (default: 32)"
    )
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)

    # Parse the arguments
    args = parser.parse_args()

    # Print out the parsed arguments
    print(f"min_snr: {args.min_snr}")
    print(f"max_snr: {args.max_snr}")
    print(f"plot={args.plot}")
    
    sim = SM_ML(SNR_dB=np.arange(args.min_snr,args.max_snr,1))
    sim.exec(plot=args.plot)