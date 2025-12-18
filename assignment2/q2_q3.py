import numpy as np
import matplotlib.pyplot as plt

def dft_vec(x):
    x = np.asarray(x, dtype=float)
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W @ x

SR = 100
ts = 1 / SR
t = np.arange(0, 1, ts)

x1 = 2.5 * np.sin(2 * np.pi * 1 * t)
x2 = 1.5 * np.sin(2 * np.pi * 3 * t)
x3 = 0.5 * np.sin(2 * np.pi * 8 * t)
x  = x1 + x2 + x3

X = dft_vec(x)

N = len(X)
n = np.arange(N)
T = N / SR
freq = n / T

mag = np.abs(X)

plt.figure(figsize=(10, 4))
plt.stem(freq, mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X[k]|")
plt.title("Two-sided DFT Magnitude Spectrum (Vectorized DFT)")
plt.grid(True)
plt.tight_layout()
plt.show()
