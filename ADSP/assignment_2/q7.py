import numpy as np
import matplotlib.pyplot as plt
from q6 import fft

SR = 128
ts = 1 / SR
t = np.arange(0, 1, ts)  # 128 samples

x1 = 2.5 * np.sin(2*np.pi*1*t)
x2 = 1.5 * np.sin(2*np.pi*3*t)
x3 = 0.5 * np.sin(2*np.pi*8*t)
x  = x1 + x2 + x3

if __name__ == '__main__':
    X = fft(x)
    N = len(X)

    # Two sided amplitude spectrum
    freq_two = np.fft.fftshift(np.fft.fftfreq(N, d=1 / SR))  # -SR/2 ... +SR/2
    mag_two = np.fft.fftshift(np.abs(X) / N)  # normalize by N

    plt.figure(figsize=(10, 4))
    plt.stem(freq_two, mag_two)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)| (normalized by N)")
    plt.title("Two-sided Amplitude Spectrum (fft, SR=128 Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # One sided amplitude spectrum (0 to Nyquist)
    half = N // 2
    freq_one = np.arange(half) * SR / N  # 0 ... SR/2-Δf
    mag_one = np.abs(X[:half]) / (N / 2)

    plt.figure(figsize=(10, 4))
    plt.stem(freq_one, mag_one)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)| (one sided, normalized by N/2)")
    plt.title("One-sided Amplitude Spectrum (fft, SR=128 Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
