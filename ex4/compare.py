#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

from ex2.STC import STC
from ex4.SM_ML import SM_ML


my_stc = STC(N_Rx=1, SNR_dB=np.arange(0, 19, 1))
my_sm_ml = SM_ML(np.arange(0, 22, 1))

stc_snr, stc_ser = my_stc.exec()
smml_snr, smml_ser = my_sm_ml.exec()

plt.semilogy(stc_snr, stc_ser, marker='o', label="STC 2x1")
plt.semilogy(smml_snr, smml_ser, marker='^', label="Spatial Multiplexing ML 2X2")
plt.grid()
plt.legend()
plt.title("STC vs SM-ML comparison")
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.show()