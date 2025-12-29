import numpy as np
import matplotlib.pyplot as plt

T = 4.0
w0 = 2*np.pi / T 

def a_k(k: int) -> float:
    """Fourier series coefficient a_k for the given square pulse."""
    if k == 0:
        return 0.5
    return np.sin(k*np.pi/2) / (k*np.pi)

def xN(t: np.ndarray, N: int) -> np.ndarray:
    """Partial sum x_N(t) = sum_{k=-N}^N a_k e^{jk w0 t} (real-valued here)."""
    ks = np.arange(-N, N+1)
    ak = np.array([a_k(int(k)) for k in ks])
    # Vectorized sum over k
    y = np.sum(ak[:, None] * np.exp(1j * ks[:, None] * w0 * t[None, :]), axis=0)
    return y.real

def x_true(t: np.ndarray) -> np.ndarray:
    """One-period definition over [-2,2): 1 for |t|<1 else 0 (periodic extension)."""
    t_mod = ((t + 2) % 4) - 2  # map to [-2,2)
    return np.where(np.abs(t_mod) < 1, 1.0, 0.0)

# Time axis over one fundamental period
t = np.linspace(-2, 2, 6000, endpoint=False)
x = x_true(t)

Ns = [1, 3, 7, 19, 43, 79]

for N in Ns:
    y = xN(t, N)

    plt.figure(figsize=(8, 3))
    plt.plot(t, y, label=f"x_N(t), N={N}")
    plt.plot(t, x, linewidth=2, alpha=0.7, label="x(t)")
    plt.xlim(-2, 2)
    plt.ylim(-0.3, 1.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title(f"Fourier Partial Sum Approximation (T=4), N={N}")
    plt.legend()
    plt.tight_layout()
    plt.show()
