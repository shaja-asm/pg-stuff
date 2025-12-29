import numpy as np
import matplotlib.pyplot as plt
from q2_q3 import dft

SR = 100
ts = 1 / SR
t = np.arange(0, 1, ts)  # 100 samples

x1 = 2.5 * np.sin(2*np.pi*1*t)
x2 = 1.5 * np.sin(2*np.pi*3*t)
x3 = 0.5 * np.sin(2*np.pi*8*t)
x  = x1 + x2 + x3

if __name__ == '__main__':

    X = dft(x)
    N = len(X)

    # Frequency axis
    n = np.arange(N)
    T = N / SR
    freq = n / T

    # One sided magnitude (first half) and normalize by N/2
    half = N // 2
    freq_one = freq[:half]
    mag_one = np.abs(X[:half]) / (N / 2)

    # Plot 1: one sided spectrum (0 to Nyquist)
    plt.figure(figsize=(10, 4))
    plt.stem(freq_one, mag_one)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X[k]| (one-sided, normalized)")
    plt.title("One-sided DFT Magnitude Spectrum (0 to Nyquist)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: zoom to 0–10 Hz
    plt.figure(figsize=(10, 4))
    plt.stem(freq_one, mag_one)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X[k]| (one-sided, normalized)")
    plt.title("One-sided DFT Magnitude Spectrum (Zoomed to 0–10 Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print the 3 strongest peaks in 0–10 Hz region
    mask = (freq_one > 0) & (freq_one <= 10)
    idx = np.where(mask)[0]
    top3 = idx[np.argsort(mag_one[idx])[-3:]][::-1]

    print("Top peaks (0–10 Hz):")
    for k in top3:
        print(f"  f = {freq_one[k]:.1f} Hz, amplitude = {mag_one[k]:.3f}")
