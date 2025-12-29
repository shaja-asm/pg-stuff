import numpy as np

def fft(x):
    x = np.asarray(x, dtype=complex)
    N = x.size

    # check power of 2
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("Length N must be a power of 2 for radix-2 fft.")

    # base case
    if N == 1:
        return x

    # split into even and odd indices
    X_even = fft(x[::2])
    X_odd  = fft(x[1::2])

    # combine
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)  # twiddle factors

    top = X_even + W * X_odd
    bot = X_even - W * X_odd

    return np.concatenate([top, bot])