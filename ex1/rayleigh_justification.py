import numpy as np
import matplotlib.pyplot as plt

# Parameters
f_c = 2.5e9  # Carrier frequency (Hz)
N = int(1e5)  # Number of iterations

# Initialize array to store transfer function values
H_f_c = np.zeros(N, dtype=complex)

# Calculate the frequency domain transfer function
for k in range(N):
    tau_vec = (50 + np.random.rand(10) * 50) / 3e8  # Delays in seconds
    H_f_c[k] = np.sum(np.exp(-1j * 2 * np.pi * f_c * tau_vec))

# Plot histograms of the real and imaginary parts of H_f_c
plt.figure()
plt.hist(H_f_c.real, bins=50)
plt.title('Real Part Distribution')

plt.figure()
plt.hist(H_f_c.imag, bins=50)
plt.title('Imaginary Part Distribution')

plt.show()

# Calculate correlation between real and imaginary parts
corr = np.sum(H_f_c.real * H_f_c.imag) / (np.sqrt(np.var(H_f_c.real)) * np.sqrt(np.var(H_f_c.imag)) * N)
print("Correlation between real and imaginary parts:", corr)
