#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))


class DoaMRC:
    def __init__(self, desired_doa, n_rx):
        # desired DOA is theta from the lectures
        self.theta = desired_doa
        self.n_rx = n_rx

    def calc_response(self, actual_doa):
        # actual DOA is phi from the lectures
        return abs(np.sum([np.exp(1j * np.pi * n * (np.sin(self.theta) - np.sin(actual_doa))) for n in range(self.n_rx)])) / self.n_rx


    def calc_response_over_range(self, actual_doas_range):
        results = []
        for actual_doa in actual_doas_range:
            response = self.calc_response(actual_doa)
            results.append(response)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Configure Simulation parameters: Desired DOA (theta), Number of RX antennas (N),\n
                       min actual DOA (phi min) max acutal DOA (phi max)"""
    )
    parser.add_argument(
        "--N_Rx",
        type=int,
        default=4,
        help="Number of RX antennas (default: 4)"
    )
    parser.add_argument(
        "--desired_doa",
        type=float,
        default=30,
        help = "Desired DOA in radians (default: pi/6 rads = 30degs)"
    )
    parser.add_argument(
        "--min_phi",
        type=float,
        default=0
    )
    parser.add_argument(
        "--max_phi",
        type=float,
        default=90
    )
    args = parser.parse_args()

    print(f"N_rx: {args.N_Rx}")
    print(f"Desired DOA: {args.desired_doa}")
    print(f"min phi: {args.min_phi}")
    print(f"max phi: {args.max_phi}")

    doa_mrc = DoaMRC(np.deg2rad(args.desired_doa), args.N_Rx)
    phi_values = np.linspace(np.deg2rad(args.min_phi), np.deg2rad(args.max_phi), 500)
    results = np.array(doa_mrc.calc_response_over_range(phi_values))
    results_dB = 20 * np.log10(results + 1e-3)  # adding 1e-3 to avoid log of zero at edges
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.rad2deg(phi_values), results_dB)
    plt.xlabel(r"$\phi$ (degs)")
    plt.ylabel(r"Response (dB)")
    plt.title(fr"MRC Response with {args.N_Rx} RX antennas and {args.desired_doa:.0f}{chr(176)} desired DOA ($\theta$)")
    plt.legend()
    plt.grid()
    plt.show()