import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks

fs = 1000  # Hz (given)
x = np.loadtxt("wavdata.csv", delimiter=",")
N = len(x)
t = np.arange(N) / fs

# Remove DC offset (good practice)
x = x - np.mean(x)

# scipy FFT + one-sided magnitude
X = fft(x)
freqs = fftfreq(N, d=1/fs)

pos = freqs >= 0
f_pos = freqs[pos]
mag = np.abs(X[pos]) / N

# One-sided scaling for real signals
if len(mag) > 2:
    mag[1:-1] *= 2

# Peak detection
# threshold = 10% of max peak
peaks, props = find_peaks(mag, height=np.max(mag) * 0.2, distance=5)

# Sort peaks by amplitude (highest first)
peak_freqs = f_pos[peaks]
peak_amps  = mag[peaks]
order = np.argsort(peak_amps)[::-1]

# keep only the top 2 peaks
top2 = order[:2]
top2_freqs = peak_freqs[top2]
top2_amps  = peak_amps[top2]

print("Top 2 detected peaks (Hz, amplitude):")
for f, a in zip(top2_freqs, top2_amps):
    print(f"  {f:.1f} Hz , {a:.6f}")

# Plot spectrum
plt.figure(figsize=(11, 4))
plt.plot(f_pos, mag)
plt.plot(top2_freqs, top2_amps, "o", label="Top peaks")
plt.title("One-sided Magnitude Spectrum of Captured Signal (SciPy FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, fs/2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
