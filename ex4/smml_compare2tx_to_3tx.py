#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

from ex4.SM_ML import SM_ML


my_sm_ml_2 = SM_ML(N_Tx=2, SNR_dB=np.arange(0, 32, 1))
my_sm_ml_3 = SM_ML(N_Tx=3, SNR_dB=np.arange(0, 32, 1))

sm_ml_2_snr, sm_ml_2_ser = my_sm_ml_2.exec()
sm_ml_3_snr, sm_ml_3_ser = my_sm_ml_3.exec()

plt.semilogy(sm_ml_2_snr, sm_ml_2_ser, marker='o', label="Spatial Multiplexing ML 2x2")
plt.semilogy(sm_ml_3_snr, sm_ml_3_ser, marker='^', label="Spatial Multiplexing ML 3X2")
plt.grid()
plt.legend()
plt.title("SM-ML 2Tx to 3Tx comparison")
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.show()