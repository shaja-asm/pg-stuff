import numpy as np
import matplotlib.pyplot as plt

fs = 2000  # Hz
T = 1.0    # seconds
t = np.arange(0, T, 1/fs)

# Three sinusoids (zero phase)
x1 = 4.0 * np.sin(2*np.pi*2*t)
x2 = 3.0 * np.sin(2*np.pi*5*t)
x3 = 0.5 * np.sin(2*np.pi*9*t)

# Combined signal
x = x1 + x2 + x3

plt.figure(figsize=(11, 4))
plt.plot(t, x1, label="x1(t) = 4 sin(2π·2t)")
plt.plot(t, x2, label="x2(t) = 3 sin(2π·5t)")
plt.plot(t, x3, label="x3(t) = 0.5 sin(2π·9t)")
plt.title("Three Individual Sinusoids (fs = 2000 Hz, 0–1 s)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Combined signal plot
plt.figure(figsize=(11, 4))
plt.plot(t, x, label="x(t) = x1 + x2 + x3")
plt.title("Combined Signal: Sum of the Three Sinusoids")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Number of samples:", len(t))
