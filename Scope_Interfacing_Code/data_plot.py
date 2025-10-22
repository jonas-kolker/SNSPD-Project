import numpy as np
import matplotlib.pyplot as plt
import scope_stuff_MDP as ss

# ref_data = np.load("C:\LeCroy\ScopeData\ReferenceWaveforms_DCcompensate0\\ref_data_000.npy")
# chip_data = np.load("C:\LeCroy\ScopeData\ChipWaveforms_DCcompensate0\\chip_data_000.npy")    

# clip = int(0)

# fig, ax = plt.subplots(2,1)
# ax[0].plot(ref_data[0][clip:],ref_data[1][clip:])
# ax[0].set_title("Ref signal")
# ax[1].plot(chip_data[0][clip:],chip_data[1][clip:])
# ax[1].set_title("Chip signal")
# print(ref_data[0].shape)
# plt.show()

# # offset_vals = ss.get_offsets(ref_data,
# #                             chip_data,
# #                             ref_threshold=.05,
# #                             chip_threshold=0.5,
# #                             clip=clip,
# #                             mismatch_handling=True,
# #                             num_samples=int(1e3))
offsets = np.loadtxt("C:\LeCroy\ScopeData\\offset_values_all_DCcompensate6.txt")
fig, hist, bins =  ss.make_histogram_and_gaussian(offsets, stdv_cutoff=3)

plt.show()