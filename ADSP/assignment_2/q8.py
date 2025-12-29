import numpy as np
import timeit
from q2_q3 import dft
from q6 import fft

def generate_sig(sr):
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)
    freq = 1.0
    x = 5 * np.sin(2*np.pi*freq*t)
    return x

def best_time(fn, repeat=5, number=1):
    return min(timeit.repeat(fn, repeat=repeat, number=number))

if __name__ == '__main__':
    for N in [1024, 4096]:
        x = generate_sig(N)

        t_dft = best_time(lambda: dft(x), repeat=3, number=1)
        t_fft = best_time(lambda: fft(x), repeat=5, number=1)

        print(f"N = {N}")
        print(f"  DFT time = {t_dft:.6f} s")
        print(f"  fft time = {t_fft:.6f} s")
        print(f"  Speedup (DFT/FFT) = {t_dft / t_fft:.2f}x")
        print("-" * 40)
