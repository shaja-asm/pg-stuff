import numpy as np

T = 4.0
w0 = 2*np.pi / T

def a_k(k: int) -> float:
    if k == 0:
        return 0.5
    return np.sin(k*np.pi/2) / (k*np.pi)

def xN(t: np.ndarray, N: int) -> np.ndarray:
    ks = np.arange(-N, N+1)
    ak = np.array([a_k(int(k)) for k in ks])
    y = np.sum(ak[:, None] * np.exp(1j * ks[:, None] * w0 * t[None, :]), axis=0)
    return y.real

Ns = [1, 3, 7, 19, 43, 79]
t = np.linspace(-2, 2, 400000, endpoint=False)

print("N   max(x_N)     overshoot%")
for N in Ns:
    y = xN(t, N)
    M = y.max()
    overshoot_percent = (M - 1.0) * 100
    print(f"{N:<3d} {M:>10.6f}   {overshoot_percent:>9.6f}")
