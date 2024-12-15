import numpy as np
import matplotlib.pyplot as plt
import os
from utils import last_two_distinct_elements

# Parameters
QPSK_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)  # QPSK symbols
SNRdB = np.arange(0, 22, 2)  # SNR in dB
SNR = 10**(SNRdB / 10)  # Convert SNR to linear scale

N = 1000000  # Number of symbols
N_Rx_values = [2, 3, 4]  # Number of RX antennas to simulate

# Plot setup
plt.figure()

for N_Rx in N_Rx_values:
    print(f"-------- {N_Rx} RX antennas ---------")
    SER = np.zeros(len(SNRdB))  # Initialize SER array for current N_Rx

    # Simulation
    for k in range(len(SNRdB)):
        print(f"{SNRdB[k]} SNRdB")
        s_hat = np.zeros(N, dtype=complex)  # Detected symbols
        rho = 1 / np.sqrt(SNR[k])  # Noise scaling factor

        # Signal generation
        s = np.random.choice(QPSK_vec, size=N)  # Random QPSK symbols
        h = (np.random.randn(N_Rx, N) + 1j * np.random.randn(N_Rx, N)) / np.sqrt(2)  # Rayleigh channel
        n = (np.random.randn(N_Rx, N) + 1j * np.random.randn(N_Rx, N)) / np.sqrt(2)  # Noise

        for kk in range(N):
            y = h[:, kk] * s[kk] + rho * n[:, kk]  # Received signal
            # Signal detection using MRC
            s_hat[kk] = np.conj(h[:, kk]) @ y / np.linalg.norm(h[:, kk])**2

        # QPSK symbol estimation
        s_tilde = (np.sign(np.real(s_hat)) + 1j * np.sign(np.imag(s_hat))) / np.sqrt(2)

        # Calculate number of errors
        NumOfErrors = np.sum(np.abs(s_tilde - s) > np.finfo(float).eps)
        SER[k] = NumOfErrors / N  # Symbol Error Rate
        print(f"SER: {SER[k]}")
    
    
    first, second = last_two_distinct_elements(SER)
    diversity_order = -(np.log10(SER[first]) - np.log10(SER[second])) / (
        SNRdB[first] - SNRdB[second]
    )
    print(f"Diversity Order for N_Rx = {N_Rx}: {10*diversity_order:.2f}")
    # Plot SER curve for current N_Rx
    plt.semilogy(SNRdB, SER, marker='o', linestyle='-', label=f'MRC Rayleigh {N_Rx} Antennas')

# Finalize plot
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('MRC Rayleigh with Multiple Antennas')
plt.legend()

# Save plot
output_dir = "MRC_plots"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "MRC_combined.png")
plt.savefig(plot_path)

plt.show()
