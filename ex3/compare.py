import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

from ex2.STC import STC
from ex3.EigenBF import EigenBF


my_stc = STC(N_Rx=2, SNR_dB=np.arange(0, 19, 1))
eigen_bf4 = EigenBF(N_Tx=4, N_Rx=2, SNR_dB=np.arange(0,8,1))
eigen_bf2 = EigenBF(N_Tx=2, N_Rx=2, SNR_dB=np.arange(0,12,1))

stc_snr, stc_ser = my_stc.exec()
eigen4_snr, eigen4_ser = eigen_bf4.exec()
eigen2_snr, eigen2_ser = eigen_bf2.exec()

plt.semilogy(stc_snr, stc_ser, marker='o', label="STC 2x2")
plt.semilogy(eigen4_snr, eigen4_ser, marker='^', label="Eigen BF 4X2")
plt.semilogy(eigen2_snr, eigen2_ser, marker='v', label="Eigen BF 2x2")
plt.grid()
plt.legend()
plt.title("STC vs Eigen BF comparison")
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.show()