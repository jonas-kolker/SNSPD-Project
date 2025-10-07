import numpy as np
import matplotlib.pyplot as plt

total_duration = 1000 # ns
drop_duration = 5 # ns
fs = 40 # GHz (samples per ns)

mean_interval1 = 50 # Avg time between signal 1 drops (ns)
std_1 = 10

mean_interval2 = 10 # Avg lag time between signal 1 and 2 drops (ns)
jitter = 5 # Stdv of lag time between signal 1 and 2 drops (ns)

n_samples = int(fs * total_duration)
signal1 = np.ones(n_samples)
signal2 = np.ones(n_samples)
jitter = .5

drop_samples = int(fs * drop_duration)
t = 0  # current time index

while t < n_samples:

    dt1 = max(0, np.random.normal(mean_interval1, std_1)) # Time between reference signal drops
    dt2 = max(0, np.random.normal(mean_interval2, jitter)) # Time between reference drop and corresponding secondary signal drop
    
    t += int(dt1 * fs)  # advance by interval (converted to samples)
    
    if t >= n_samples:
        break

    # Apply drop to signal 1
    signal1[t : min(t + drop_samples, n_samples)] = 0

    # Advance by dt2 interval
    t += int(dt2*fs)

    # Apply delayed drop to signal 2
    signal2[t: min(t + drop_samples, n_samples)] = 0

    # Move past the dropped samples
    t += drop_samples

time = np.linspace(0, total_duration, total_duration*fs)

signal1_array = np.vstack((time, signal1))
signal2_array = np.vstack((time, signal2))

plt.plot(signal1_array[0], signal1_array[1], label="Reference")
plt.plot(signal2_array[0], signal2_array[1], label="Chip")
plt.title("Simulated Signals")
plt.legend()
plt.show()

# np.save("test_ref_sig.npy", signal1_array)
# np.save("test_chip_sig.npy", signal2_array)