import numpy as np
import matplotlib.pyplot as plt
import scope_script_MDP as ss
import os

dir = os.getcwd()

trigger_data = np.load(dir + "/test_ref_sig.npy")
chip_data = np.load(dir + "/test_chip_sig.npy")

# plt.plot(trigger_data[0], trigger_data[1], label="Trigger Signal")
# plt.plot(chip_data[0], chip_data[1], label="Chip Signal")
# plt.legend()
# plt.show()

offset_vals = ss.get_offsets(trigger_data, chip_data)
hist, bin_edges, popt, perr = ss.make_historgram_and_guassian(offset_vals)