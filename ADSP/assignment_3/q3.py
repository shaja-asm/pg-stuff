import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.fftpack import fft, ifft, fftfreq

fs = 2000
T = 1.0
t = np.arange(0, T, 1/fs)
N = len(t)

x1 = 4.0 * np.sin(2*np.pi*2*t)
x2 = 3.0 * np.sin(2*np.pi*5*t)
x3 = 0.5 * np.sin(2*np.pi*9*t)
x = x1 + x2 + x3

# scipy FFT + frequency axis
Xs = fft(x)
freqs_s = fftfreq(N, d=1/fs)

# One-sided amplitude spectrum (scaled)
pos_mask = freqs_s >= 0
f_pos = freqs_s[pos_mask]
Xs_pos = Xs[pos_mask]

A_pos = np.abs(Xs_pos) / N
if len(A_pos) > 2:
    A_pos[1:-1] *= 2  # double except DC and Nyquist

# scipy IFFT reconstruction
x_rec = ifft(Xs).real
max_err = np.max(np.abs(x - x_rec))

# Plot FFT amplitude spectrum
fmax_plot = 50
idx = f_pos <= fmax_plot

plt.figure(figsize=(11, 4))
plt.stem(f_pos[idx], A_pos[idx])
plt.title("One-sided FFT Amplitude Spectrum (SciPy fftpack)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot original vs reconstructed
plt.figure(figsize=(11, 4))
plt.plot(t, x, label="Original x(t)")
plt.plot(t, x_rec, "--", label="Reconstructed (SciPy IFFT)")
plt.title(f"Original vs SciPy IFFT Reconstruction (max error = {max_err:.3e})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Timing comparison of numpy vs scipy
num_runs = 2000

t_numpy_total = timeit.timeit("np.fft.fft(x)", globals=globals(), number=num_runs)
t_scipy_total = timeit.timeit("fft(x)", globals=globals(), number=num_runs)

t_numpy_avg = t_numpy_total / num_runs
t_scipy_avg = t_scipy_total / num_runs

print(f"N = {N} samples, fs = {fs} Hz")
print("Timing over", num_runs, "runs:")
print(f"  NumPy FFT  total = {t_numpy_total:.6f} s, avg = {t_numpy_avg*1e6:.3f} microseconds")
print(f"  SciPy FFT  total = {t_scipy_total:.6f} s, avg = {t_scipy_avg*1e6:.3f} microseconds")

speedup = t_numpy_avg / t_scipy_avg if t_scipy_avg > 0 else np.nan
print(f"  Speed ratio (NumPy/SciPy) = {speedup:.2f}x")
