import numpy as np
import matplotlib.pyplot as plt
import scope_stuff_MDP as ss

# ref_data = np.load("C:\LeCroy\ScopeData\ReferenceWaveforms_DCcompensate2\\ref_data_000.npy")
# chip_data = np.load("C:\LeCroy\ScopeData\ChipWaveforms_DCcompensate2\\chip_data_000.npy")    

# fig, ax = plt.subplots(2,1)
# ax[0].plot(ref_data[0][10000:],ref_data[1][10000:])
# ax[0].set_title("Ref signal")
# ax[1].plot(chip_data[0][10000:],chip_data[1][10000:])
# ax[1].set_title("Chip signal")

# plt.show()

offsets = np.loadtxt("C:\LeCroy\ScopeData\\offset_values_all_DCcompensate2.txt")
fig, hist, bins =  ss.make_histogram_and_gaussian(offsets, stdv_cutoff=4)

plt.show()