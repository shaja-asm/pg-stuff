import numpy as np
import matplotlib.pyplot as plt


fs = 1000  # Hz
data = np.loadtxt("wavdata.csv", delimiter=",")

# Ensure shape is (N, C)
if data.ndim == 1:
    data = data.reshape(-1, 1)   # mono -> (N,1)

N, C = data.shape
t = np.arange(N) / fs

print(f"Samples N = {N}, Channels = {C}, Duration = {N/fs:.2f} s")

# Plot full time series
plt.figure(figsize=(11, 4))
for ch in range(C):
    plt.plot(t, data[:, ch], label=f"Channel {ch+1}")
plt.title("Captured signal (time series) - full duration")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Zoomed plot
t_zoom = 0.1  # seconds
Nz = int(t_zoom * fs)

plt.figure(figsize=(11, 4))
for ch in range(C):
    plt.plot(t[:Nz], data[:Nz, ch], label=f"Channel {ch+1}")
plt.title(f"Captured signal (zoomed) - first {t_zoom} s")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
