import numpy as np
import matplotlib.pyplot as plt
import timeit

fs = 2000  # Hz
T = 1.0    # seconds
t = np.arange(0, T, 1/fs)
N = len(t)

x1 = 4.0 * np.sin(2*np.pi*2*t)
x2 = 3.0 * np.sin(2*np.pi*5*t)
x3 = 0.5 * np.sin(2*np.pi*9*t)
x = x1 + x2 + x3

# FFT + frequency axis
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)

# Amplitude spectrum (two-sided, scaled)
mag_two_sided = np.abs(X) / N

# One-sided amplitude spectrum
pos_mask = freqs >= 0
f_pos = freqs[pos_mask]
X_pos = X[pos_mask]

A_pos = np.abs(X_pos) / N
if len(A_pos) > 2:
    A_pos[1:-1] *= 2  # double everything except DC and Nyquist

# IFFT reconstruction
x_rec = np.fft.ifft(X).real
max_err = np.max(np.abs(x - x_rec))

# Plot FFT amplitude spectrum (stem)
fmax_plot = 50
idx = f_pos <= fmax_plot

plt.figure(figsize=(11, 4))
markerline, stemlines, baseline = plt.stem(f_pos[idx], A_pos[idx])
plt.title("One-sided FFT Amplitude Spectrum (NumPy FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot original vs reconstructed (time domain)
plt.figure(figsize=(11, 4))
plt.plot(t, x, label="Original x(t)")
plt.plot(t, x_rec, "--", label="Reconstructed (IFFT)")
plt.title(f"Original vs IFFT Reconstruction (max error = {max_err:.3e})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Timing FFT using timeit.
num_runs = 2000
total_time = timeit.timeit("np.fft.fft(x)", globals=globals(), number=num_runs)
avg_time = total_time / num_runs

print(f"N = {N} samples, fs = {fs} Hz")
print(f"FFT timing over {num_runs} runs:")
print(f"  Total time = {total_time:.6f} s")
print(f"  Avg per fft = {avg_time*1e6:.3f} microseconds")
