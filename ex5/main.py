import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_rx = 2

QPSK_vec = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

SNRdB = np.arange(0, 43, 2) if N_rx < 3 else np.arange(0, 31, 2)
SNR = 10 ** (SNRdB / 10)

SER_ZF = np.zeros(len(SNRdB))
SER_ML = np.zeros(len(SNRdB))
N = 50000  # Number of pairs per SNR point

# Create a Matrix of 16 combinations for ML
CombinationLoc = np.zeros((2, 16), dtype=int)
CombinationLoc[0, :] = np.floor(np.arange(16) / 4).astype(int)
CombinationLoc[1, :] = np.mod(np.arange(16), 4).astype(int)

CombinationMat = QPSK_vec[CombinationLoc]

for k in range(len(SNRdB)):
    s_hat_ZF = np.zeros((2, N), dtype=complex)
    s_tilde_ML = np.zeros((2, N), dtype=complex)
    
    rho = 1 / np.sqrt(SNR[k])
    # Signal Generation
    s = QPSK_vec[np.random.randint(0, 4, (2, N))]
    n = (np.random.randn(N_rx, N) + 1j * np.random.randn(N_rx, N)) / np.sqrt(2)
    
    for kk in range(N):
        # Signal generation
        H = (np.random.randn(N_rx, 2) + 1j * np.random.randn(N_rx, 2)) / np.sqrt(2)
        
        y = H @ (1 / np.sqrt(2)) * s[:, kk] + rho * n[:, kk]
        
        # Signal Detection ZF
        s_hat_ZF[:, kk] = np.linalg.pinv(1 / np.sqrt(2) * H) @ y
        
        # Signal Detection ML
        DiffVec = np.tile(y, (16, 1)).T - 1 / np.sqrt(2) * H @ CombinationMat
        CostVec = np.sum(np.abs(DiffVec) ** 2, axis=0)
        
        MinLoc = np.argmin(CostVec)
        s_tilde_ML[:, kk] = CombinationMat[:, MinLoc]
    
    s_tilde_ZF = np.sign(np.real(s_hat_ZF)) / np.sqrt(2) + 1j * np.sign(np.imag(s_hat_ZF)) / np.sqrt(2)
    
    NumOfErrorsZF = np.sum(np.abs(s_tilde_ZF - s) > np.finfo(float).eps)
    SER_ZF[k] = NumOfErrorsZF / N / 2
    
    NumOfErrorsML = np.sum(np.abs(s_tilde_ML - s) > np.finfo(float).eps)
    SER_ML[k] = NumOfErrorsML / N / 2

# Plotting
plt.semilogy(SNRdB, SER_ZF, label='ZF')
plt.grid(True)
plt.semilogy(SNRdB, SER_ML, 'r', label='ML')
plt.xlabel('SNR(dB)')
plt.ylabel('SER')
plt.title(f'SM {N_rx}X2 Rayleigh')
plt.legend()
plt.show()
