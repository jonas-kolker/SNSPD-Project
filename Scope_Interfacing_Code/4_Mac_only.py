import numpy as np
import matplotlib.pyplot as plt
import scope_script_MDP as ss

filename = "/Users/jonas/Downloads/offset_values_all.txt"

offset_vals = np.loadtxt(filename)

mean = np.mean(offset_vals)
stdv = np.std(offset_vals)

# print( list( dict.fromkeys(offset_vals)))

ss.make_histogram_and_gaussian(offset_vals, stdv_cutoff=8)
