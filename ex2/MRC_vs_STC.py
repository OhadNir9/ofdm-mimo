import numpy as np
import matplotlib.pyplot as plt

# Parameters
QPSK_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)  # QPSK symbols
SNRdB = np.arange(0, 22, 2)  # SNR in dB
SNR = 10**(SNRdB / 10)  # Convert SNR to linear scale

SER = np.zeros(len(SNRdB))  # Initialize SER array
N = 100000  # Number of pairs per SNR point

# Simulation
for k in range(len(SNRdB)):
    s_hat = np.zeros((2, N), dtype=complex)  # Detected symbols
    rho = 1 / np.sqrt(SNR[k])  # Noise scaling factor

    # Signal generation
    s = np.random.choice(QPSK_vec, size=(2, N))  # Random QPSK symbols (2xN)
    h = (np.random.randn(2, N) + 1j * np.random.randn(2, N)) / np.sqrt(2)  # Rayleigh channel
    n = (np.random.randn(2, N) + 1j * np.random.randn(2, N)) / np.sqrt(2)  # Noise

    for kk in range(N):
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

    # QPSK symbol estimation
    s_tilde = (np.sign(np.real(s_hat)) + 1j * np.sign(np.imag(s_hat))) / np.sqrt(2)

    # Calculate number of errors
    NumOfErrors = np.sum(np.abs(s_tilde - s) > np.finfo(float).eps)
    SER[k] = NumOfErrors / (2 * N)  # Symbol Error Rate

# Plot results
plt.figure()
plt.semilogy(SNRdB, SER, marker='o', linestyle='-', label='STC 2x1 Rayleigh')
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('STC 2x1 Rayleigh')
plt.legend()
plt.show()
