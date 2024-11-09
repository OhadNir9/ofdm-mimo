import numpy as np
import matplotlib.pyplot as plt

# Define QPSK constellation points
QPSK_vec = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# SNR in dB and linear scale
SNRdB = np.arange(0, 31, 2)
SNR = 10 ** (SNRdB / 10)

# Initialize SER array
SER_rayligh = np.zeros(len(SNRdB))
SER_awgn = np.zeros(len(SNRdB))
N = 500000  # Number of symbols

for is_rayligh in [False, True]:
    # SER calculation for each SNR value
    for k in range(len(SNRdB)):
        rho = 1 / np.sqrt(SNR[k])

        # Signal Generation
        s = np.random.choice(QPSK_vec, N)
        if is_rayligh:
            h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        else:
            h = np.ones(N)
        
        n = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        y = h * s + rho * n

        # Signal Detection
        s_hat = y / h
        s_tilde = np.sign(np.real(s_hat)) / np.sqrt(2) + 1j * np.sign(np.imag(s_hat)) / np.sqrt(2)

        # Error calculation
        NumOfErrors = np.sum(np.abs(s_tilde - s) > np.finfo(float).eps)
        if is_rayligh:
            SER_rayligh[k] = NumOfErrors / N
        else:
            SER_awgn[k] = NumOfErrors / N

# Plotting
plt.semilogy(SNRdB, SER_rayligh, marker='x', label='Rayleigh')
plt.semilogy(SNRdB, SER_awgn, marker='o', label='AWGN')
plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('SISO Symbol Error Rate - Rayleigh and AWGN channels')
plt.legend()
plt.show()