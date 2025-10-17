import numpy as np
import matplotlib.pyplot as plt

ref_data = np.load("C:\LeCroy\ScopeData\ReferenceWaveforms\\ref_data_000.npy")
chip_data = np.load("C:\LeCroy\ScopeData\ChipWaveforms\\chip_data_000.npy")    

fig, ax = plt.subplots(2,1)
ax[0].plot(ref_data[0],ref_data[1])
ax[0].set_title("Ref signal")
ax[1].plot(chip_data[0],chip_data[1])
ax[1].set_title("Chip signal")

plt.show()
