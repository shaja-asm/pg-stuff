import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft


fs = 1000  # Hz
x = np.loadtxt("wavdata.csv", delimiter=",")

N = len(x)
x = x - np.mean(x)  # remove DC

# FFT of original
X = fft(x)
freqs = fftfreq(N, d=1/fs)

# Apply band-pass in FFT domain (320–370 Hz)
f1, f2 = 320, 370
H = ((np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)).astype(float)
Xf = X * H

# One-sided amplitude spectra (0 to fs/2)
pos = freqs >= 0
f_pos = freqs[pos]

A_orig = np.abs(X[pos]) / N
A_filt = np.abs(Xf[pos]) / N

# One-sided scaling for real signals
if len(A_orig) > 2:
    A_orig[1:-1] *= 2
    A_filt[1:-1] *= 2


# Side by side plots
plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
plt.plot(f_pos, A_orig)
plt.title("Original Signal: One-sided FFT Amplitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, fs/2)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_pos, A_filt)
plt.title("Filtered (320–370 Hz): One-sided FFT Amplitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, fs/2)
plt.grid(True)

plt.tight_layout()
plt.show()
