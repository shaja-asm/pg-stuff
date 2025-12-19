import numpy as np
import matplotlib.pyplot as plt

SR = 100  # Hz
ts = 1 / SR
t = np.arange(0, 1, ts)

x1 = 2.5 * np.sin(2 * np.pi * 1 * t)   # 1 Hz
x2 = 1.5 * np.sin(2 * np.pi * 3 * t)   # 3 Hz
x3 = 0.5 * np.sin(2 * np.pi * 8 * t)   # 8 Hz

# Sum
x = x1 + x2 + x3

if __name__ == '__main__':
    print("Number of samples =", len(t))

    plt.figure(figsize=(10, 5))
    plt.plot(t, x1, label="x1(t) = 2.5 sin(2pi·1t)")
    plt.plot(t, x2, label="x2(t) = 1.5 sin(2pi·3t)")
    plt.plot(t, x3, label="x3(t) = 0.5 sin(2pi·8t)")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Individual sinusoids and their sum (SR = 100 Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
