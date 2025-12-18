import numpy as np
import timeit

def generate_sig(sr):
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)     # 1 second duration  -> N = sr samples
    freq = 1.0
    x = 5 * np.sin(2*np.pi*freq*t)
    return x

def dft_vec(x):
    x = np.asarray(x, dtype=float)
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W @ x

for sr in [1000, 4000]:
    x = generate_sig(sr)
    N = len(x)

    t_sec = timeit.timeit(lambda: dft_vec(x), number=1)

    print(f"SR = {sr} Hz  -> N = {N} samples, DFT time = {t_sec:.6f} s")
