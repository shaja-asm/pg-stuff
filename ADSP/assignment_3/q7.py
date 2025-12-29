import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq


fs = 1000  # Hz
x = np.loadtxt("wavdata.csv", delimiter=",")

N = len(x)
t = np.arange(N) / fs

# Remove DC offset
x = x - np.mean(x)

# FFT
X = fft(x)
freqs = fftfreq(N, d=1/fs)


# Ideal band-pass mask: 320–370 Hz
f1, f2 = 320, 370
H = ((np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)).astype(float)

X_filt = X * H

# IFFT back to time domain
x_filt = ifft(X_filt).real

# One-sided amplitude spectrum for plotting
pos = freqs >= 0
f_pos = freqs[pos]

mag_orig = np.abs(X[pos]) / N
mag_filt = np.abs(X_filt[pos]) / N

# One-sided scaling (real signal)
if len(mag_orig) > 2:
    mag_orig[1:-1] *= 2
    mag_filt[1:-1] *= 2

# Plot spectra (before vs after)
plt.figure(figsize=(11, 4))
plt.plot(f_pos, mag_orig, label="Original spectrum")
plt.plot(f_pos, mag_filt, label="Filtered spectrum (320–370 Hz)")
plt.xlim(0, fs/2)  # 0–500 Hz
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Magnitude Spectrum: Before vs After Band-pass Filtering")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot time-domain: full duration
plt.figure(figsize=(11, 4))
# plt.plot(t, x, label="Original")
plt.plot(t, x_filt, label="Filtered (320–370 Hz)", alpha=0.9)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Domain: Original vs Filtered")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot time-domain: zoom around strongest filtered activity
idx_max = np.argmax(np.abs(x_filt))
win = int(0.1 * fs)  # 0.1 s window
start = max(0, idx_max - win//2)
end = min(N, start + win)

plt.figure(figsize=(11, 4))
plt.plot(t[start:end], x[start:end], label="Original (zoom)")
plt.plot(t[start:end], x_filt[start:end], label="Filtered (zoom)", alpha=0.9)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Zoomed Segment (around strongest filtered portion)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
