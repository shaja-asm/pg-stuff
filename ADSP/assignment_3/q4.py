import numpy as np

fs = 1000
data = np.loadtxt("wavdata.csv", delimiter=",")
data = np.atleast_2d(data)
if data.shape[0] == 1:      # if it became (1, N), transpose it
    data = data.T

print("Loaded shape:", data.shape)
duration = data.shape[0] / fs
print("Duration (s):", duration)
