import numpy as np
import matplotlib.pyplot as plt
import scope_stuff_MDP as ss

ref_data = np.load("C:\LeCroy\ScopeData\ReferenceWaveforms_test\\ref_data_000.npy")
chip_data = np.load("C:\LeCroy\ScopeData\ChipWaveforms_test\\chip_data_000.npy")    

num_samples= int(1e3)

print(len(ref_data[0]))

fig, ax = plt.subplots(2,1)
ax[0].plot(ref_data[1])
ax[0].set_title("Ref signal")
ax[1].plot(chip_data[1])
ax[1].set_title("Chip signal")
print(ref_data[0].shape)
plt.show()

# offset_vals = ss.get_offsets(ref_data,
#                             chip_data,
#                             ref_threshold=.05,
#                             chip_threshold=0.5,
#                             clip=clip,
#                             mismatch_handling=True,
#                             num_samples=int(5e3))

# offset_vals = np.loadtxt("C:\LeCroy\ScopeData\\offset_values_all_test.txt")
# fig, hist, bins =  ss.make_histogram_and_gaussian(offset_vals, 
#                                                   stdv_cutoff=0, 
#                                                   hist_bins=40,
#                                                   plot=True)

plt.show()

# print(ref_data[1][:10])