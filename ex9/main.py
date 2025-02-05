#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

from enum import Enum
class ChannelType(Enum):
    IDENTITY_CHANNEL = 0
    DELAY_SPREAD_CHANNEL = 1

class ChannelEstimationType(Enum):
    BY_PILOT_SYMBOL = 0
    PERFECT_KNOWLEDGE = 1

class OFDM_Ex9:
    def __init__(self, n_subcarriers=256, n_timeslots=21, cp_length=16, snr_db=20, channel_type=ChannelType.IDENTITY_CHANNEL, channel_estimation_type=ChannelEstimationType.BY_PILOT_SYMBOL):
        self.n_sc = n_subcarriers
        self.n_ts = n_timeslots
        self.cp_length = cp_length
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.channel_estimation_type = channel_estimation_type

    def _generate_channel(self):
        # Generates the discrete channel frequency domain samples
        # Add here more channel types if desired
        # Currently supports channel types from the HW requirements
        h = np.zeros(self.cp_length, dtype=complex)

        if self.channel_type == ChannelType.IDENTITY_CHANNEL:
            h[0] = 1
        
        elif self.channel_type == ChannelType.DELAY_SPREAD_CHANNEL:
            if self.cp_length < 10:
                print(f"L_CP: {self.cp_length} smaller then required for channel type: {self.channel_type}")
                exit(1)
            h[0] = 1
            h[9] = 0.9j

        else:
            print(f"invalid channel type: {self.channel_type}")
            exit(1)

        return h
        
    def _generate_transmitted_symbols(self, debug_plot=False):
        # Generate a matrix of QAM-4 symbols.
        # Each column represents a single timeslot
        # Each row represents a single subcarrier
        # So in the (x,y) cell, you'll see the symbol sent over the x subcarrier, at the y timeslot
        # These are the 'a_k's from the lecture (dk's after ifftshift)
        a = (np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=(self.n_sc, self.n_ts)) / np.sqrt(2))

        if debug_plot:
            # used to plot the trasmitted symbols
            self._plot(a[:, 1:])

        return a

    def _ofdm_modulator(self, qam_symbols):
        # This function gets the symbols we want to transmit (a_k's), and modulates
        # them to OFDM symbols.
        # The modulation is done per timeslot (ifft over each column)
        # So each returned column will be the timedomain samples of a single ofdm symbol
        ofdm_symbols = np.fft.ifft(qam_symbols, axis=0)
        
        # The next step takes the last L_CP symbols, and put them also at the beginning
        # (adds L_CP timedomain samples for each timeslot)
        ofdm_symbols_cp = np.vstack([ofdm_symbols[-self.cp_length:, :], ofdm_symbols])
        ofdm_symbols_cp_flattened = ofdm_symbols_cp.flatten(order='F')
        return ofdm_symbols_cp_flattened
    
    def _pass_through_channel(self, ofdm_symbols_cp_flattened, h):
        rx_signal = np.convolve(ofdm_symbols_cp_flattened, h, mode='full')
        # Now the shape of rx_signal is: (self.n_sc + self.cp_length) * self.n_ts + self.cp_length
        # where the last addition is a convolution artifact that we want to trim (unsure why actually)
        rx_signal = rx_signal[:len(ofdm_symbols_cp_flattened)]
        
        # Add noise
        signal_power = np.mean(np.abs(rx_signal)**2)
        noise_power = signal_power / (10**(self.snr_db / 10))
        noise_vector = ((np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal))) / np.sqrt(2)) * np.sqrt(noise_power)
        rx_signal_noised = rx_signal + noise_vector
        return rx_signal_noised
    
    def _ofdm_receiver(self, rx_signal_noised):
        # Use order 'F' to correctly reshape the samples
        # It'll take the first (n_sc + cp_length) samples to the first column (first timeslot)
        # before moving on to the next timeslot
        reshaped_rx_signal = rx_signal_noised.reshape((self.n_sc + self.cp_length, self.n_ts), order='F')
        # Now getting rid of the Cyclic Prefix
        rx_signal_no_cp = reshaped_rx_signal[self.cp_length:, :]

        # Now each column is simply the timedomain samples of a single (noised) ofdm symbol
        # We want to extract the n_sc different QAM-4 symbols from it (modulated on the different subcarriers)
        # This will be done by inverse operation to the modulation - fft!
        decoded_a = np.fft.fft(rx_signal_no_cp, axis=0)

        return decoded_a

    def _calc_channel_estimation(self, orig_a, decoded_a, h):
        if self.channel_estimation_type == ChannelEstimationType.BY_PILOT_SYMBOL:
            # That means the first symbol is pilot, followed by (n_ts -1) payload symbols
            # Estimating the frequency response over all the subcarriers
            # Assuming it's fixed in time along the payload transmission
            # (Actual validity of this method is determined by the coherence time / delay spread in the system)
            channel_estimation = decoded_a[:, 0] / orig_a[:, 0]
        elif self.channel_estimation_type == ChannelEstimationType.PERFECT_KNOWLEDGE:
            # That means perfect channel estimation
            channel_estimation = np.fft.fft(h, self.n_sc)
        
        else:
            print(f"invalid channel estimation type: {self.channel_estimation_type}")
            exit(1)
        
        return channel_estimation

    def _equalize_qam_symbols(self, decoded_a, channel_estimation_vector):
        # first timeslot is always pilot, even in the perfect estimation scenario
        payload = decoded_a[:, 1:]
        equalized_payload = payload / np.tile(channel_estimation_vector[:, None], (1, self.n_ts-1))
        return equalized_payload

    def _compute_evm(self, orig_payload, decoded_equalized_payload):
        evm = 10 * np.log10(np.mean(np.abs(decoded_equalized_payload.flatten(order='F') - orig_payload.flatten(order='F'))**2))
        return evm
    
    def _plot(self, decoded_equalized_payload, evm):
        plt.figure(figsize=(8,8))
        plt.plot(decoded_equalized_payload.flatten(order='F').real, decoded_equalized_payload.flatten(order='F').imag, '.')
        plt.grid()
        title = f"""Demodulated QAM symbols - constellation space\n
                SNR: {self.snr_db} dB, Subcarriers: {self.n_sc}, Payload Symbols: {self.n_ts-1}\n
                ChannelType: {self.channel_type.name}, ChannelEstimationType: {self.channel_estimation_type.name}\n
                EVM = {evm:.1f} dB
                """
        plt.title(title, fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def exec(self, plot=False):
        orig_qam_symbols = self._generate_transmitted_symbols(debug_plot=False)
        orig_qam_payload = orig_qam_symbols[:, 1:]
        ofdm_symbols = self._ofdm_modulator(orig_qam_symbols)
        h = self._generate_channel()
        ofdm_symbols_after_channel = self._pass_through_channel(ofdm_symbols, h)
        decoded_qam_symbols = self._ofdm_receiver(ofdm_symbols_after_channel)
        channel_estimation = self._calc_channel_estimation(orig_qam_symbols, decoded_qam_symbols, h)
        equalized_qam_payload = self._equalize_qam_symbols(decoded_qam_symbols, channel_estimation)
        evm = self._compute_evm(orig_qam_payload, equalized_qam_payload)
        print(f"Channel Type: {self.channel_type.name}")
        print(f"Channel Estimation Type: {self.channel_estimation_type.name}")
        print(f"Subcarriers: {self.n_sc}, payload symbols: {self.n_ts-1}")
        print(f"SNR: {self.snr_db} [dB]")
        print(f"Resultant EVM: {evm} [dB]")

        if plot:
            self._plot(equalized_qam_payload, evm)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="""Configure Simulation parameters"""
    )
    parser.add_argument(
        "--channel-type",
        type=int,
        default=0,
        help="Channel Type (0 for Identity, 1 for Delay Spread)"
    )
    parser.add_argument(
        "--estimation-type",
        type=float,
        default=0,
        help = "Channel Estimation Type (0 for by-pilot, 1 for by-perfect-knowledge)"
    )
    parser.add_argument(
        "--n-subcarriers",
        type=float,
        default=256
    )
    parser.add_argument(
        "--n-payload-symbols",
        type=float,
        default=20
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=20,
        help = "SNR in dB"
    )
    parser.add_argument(
        "--cp-length",
        type=int,
        default=16,
        help = "Cyclic Prefix (and channel) length (default 16)"
    )
    args = parser.parse_args()
    
    sim = OFDM_Ex9(n_subcarriers=args.n_subcarriers, n_timeslots=args.n_payload_symbols+1, cp_length=args.cp_length, snr_db=args.snr, channel_type=ChannelType(args.channel_type),
                   channel_estimation_type=ChannelEstimationType(args.estimation_type))
    sim.exec(plot=True)

