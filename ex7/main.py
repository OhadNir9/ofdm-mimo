import numpy as np
import matplotlib.pyplot as plt

class WirelessChannelModel:
    def __init__(self, f_max, fir_length=4096):
        self.f_max = f_max
        self.fir_length = fir_length

        self.sampling_freq = 40 * self.f_max  # oversampling

        # ---------- Construct the Jakes PSD -----------
        FreqAxis = np.fft.fftfreq(self.fir_length, 1 / self.sampling_freq)
        jakes_psd_nonzero_range = np.abs(FreqAxis) <= f_max  # Limit FreqAxis to the range -f_max to f_max
        print(FreqAxis)
        # That is the Jakes PSD - we want our generated signal PSD to look exactly like that.
        # We evaluate it in every point on the freq axis
        jakes_psd = np.zeros_like(FreqAxis)
        jakes_psd[jakes_psd_nonzero_range] = np.real(1 / f_max / np.sqrt(1 - (FreqAxis[jakes_psd_nonzero_range] / f_max) ** 2))
        jakes_psd[np.isinf(jakes_psd)] = 0
        #plt.figure()
        #plt.scatter(FreqAxis, jakes_psd)
        #plt.show()

        # ----------- The shaping filter ------------
        shaping_filter_freq_domain = jakes_psd ** 0.5
        shaping_filter = np.fft.ifftshift(np.fft.ifft(shaping_filter_freq_domain))  # unshift needed from the weird element order of the fft func
        shaping_filter = shaping_filter / np.linalg.norm(shaping_filter)  # Normalize for unit variance
        
        #
        ## Plot the Frequency Response G(f)
        plt.figure()
        plt.plot(np.linspace(0, self.sampling_freq - self.sampling_freq / len(shaping_filter_freq_domain), len(shaping_filter_freq_domain)), shaping_filter_freq_domain)
        plt.grid(True)
        plt.title("The Frequency Response G(f)")
        plt.xlabel("Frequency (Hz)")
        plt.show()

        # Plot the Time Domain Filter
        plt.figure()
        plt.stem(np.real(shaping_filter))
        plt.grid(True)
        plt.title("The Time Domain Filter")
        plt.show()

        #
        # Generate the Fading Processes
        channelA_path_powers_db = np.array([0, -1, -9, -10, -15, -20])
        channelA_path_powers_linear = 10 ** (channelA_path_powers_db / 20)
        channelA_path_delays_secs = np.array([0, 310, 710, 1090, 1730, 2510]) * 1e-9
        
        #
        n_samples = 100000
        n_samples_with_guards = n_samples + self.fir_length - 1
        
        #
        dest_random_process = np.zeros((len(channelA_path_powers_db), n_samples), dtype=complex)

        #
        for path in range(len(channelA_path_powers_db)):
            white_noise = (np.random.randn(n_samples_with_guards) + 1j * np.random.randn(n_samples_with_guards)) / np.sqrt(2)
            white_noise_after_shaping_filter = np.convolve(white_noise, shaping_filter, mode="full")
            dest_random_process[path, :] = white_noise_after_shaping_filter[self.fir_length - 1:n_samples + self.fir_length - 1] * channelA_path_powers_linear[path]  # Add power to each process
        
        #
        coherence_time = 1 / (5 * f_max)
        
        short_period_secs = coherence_time / 8
        short_period_samples = int(np.fix(short_period_secs * self.sampling_freq))
        long_period_secs = coherence_time * 10
        long_period_samples = int(np.fix(long_period_secs * self.sampling_freq))

        #
        # Compute the Channel Responses
        required_samples = [0, short_period_samples, long_period_samples]
        
        freq_range = np.linspace(-10e6, 10e6, 2001)
        freq_responses = np.zeros((len(required_samples), len(freq_range)), dtype=complex)
        
        for k, sample_idx in enumerate(required_samples):
            for kk in range(len(channelA_path_powers_db)):
                freq_responses[k, :] += dest_random_process[kk, sample_idx] * np.exp(-1j * 2 * np.pi * channelA_path_delays_secs[kk] * freq_range)
        
        ## Plot the Frequency Responses
        plt.figure()
        plt.plot(freq_range, np.abs(freq_responses).T)
        plt.grid(True)
        plt.legend(['t_0', 't_0+\u0394_1, \u0394_1<<T_c', 't_0+\u0394_2, \u0394_2>>T_c'])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|H(f)|")
        plt.show()
        #

wls_channel = WirelessChannelModel(f_max=2.5e2)
