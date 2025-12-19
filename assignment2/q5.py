import numpy as np
import timeit
from q2_q3 import dft

def generate_sig(sr):
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)
    freq = 1.0
    x = 5 * np.sin(2*np.pi*freq*t)
    return x

if __name__ == '__main__':
    for sr in [1000, 4000]:
        x = generate_sig(sr)
        N = len(x)

        t_sec = timeit.timeit(lambda: dft(x), number=1)

        print(f"SR = {sr} Hz  -> N = {N} samples, DFT time = {t_sec:.6f} s")
