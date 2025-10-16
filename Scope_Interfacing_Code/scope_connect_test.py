import win32com.client #imports the pywin32 library
import scope_script_MDP as ss
from MAUI import MAUI
import matplotlib.pyplot as plt
import time
import numpy as np
import os, gc

if __name__=="__main__":
    
    ref_array_list = []
    chip_array_list = []
    offset_vals_list = []
    
    N = 20
    num_samples = int(1e4) 
    div_time = 50-9 # There are 10 divisons per acquisition
    
    num_loops = 2
    ref_thresh = .08
    chip_thresh = 0.00
    clip = num_samples
    
    # Make appropriate (sub)directories for storing data from each loop
    save_dir = "C:\\LeCroy\\ScopeData"
    save_dir_ref = save_dir + "\\ReferenceWaveforms"
    save_dir_chip = save_dir + "\\ChipWaveforms"
    save_dir_offset = save_dir +"\\OffsetVals"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_ref, exist_ok=True)
    os.makedirs(save_dir_chip, exist_ok=True)
    os.makedirs(save_dir_offset, exist_ok=True)

    # Create final files that will hold "all" the data
    combined_offset_file = os.path.join(save_dir, "offset_values_all.txt")
    combined_ref_data_file = os.path.join(save_dir, "ref_data_net.npy")
    combined_chip_data_file = os.path.join(save_dir, "chip_data_net.npy")
    
    # Remove previous data if it's still there
    if os.path.exists(combined_offset_file):
        os.remove(combined_offset_file)
    if os.path.exists(combined_ref_data_file):
        os.remove(combined_ref_data_file)
    if os.path.exists(combined_chip_data_file):
        os.remove(combined_chip_data_file)

    with MAUI() as c:
        loop = 0
        
        while loop < num_loops:
            print(f"Loop {loop}")
            # Reset scope settings
            c.reset()

            # Set appropriate voltage scales and time divisions for acquisition
            c.set_vertical_scale("C1", .05)# For AWG this should be 20 mV (.020)
            c.set_vertical_scale("C2", .22)

            c.idn() # Needed for code to run for some reason

            # Create files to save waveforms from this loop
            ref_data_file_i = os.path.join(save_dir_ref, f"ref_data_{loop:03}.npy")
            chip_data_file_i = os.path.join(save_dir_chip, f"chip_data_{loop:03}.npy")
            offset_file_i = os.path.join(save_dir_offset, f"offset_values_{loop:03}.txt")

            # Get the absolute time wrt previous loops so that time data between files is distinct
            time_this_loop = N*10*div_time*loop # Number of waveforms * 10 time divisions per waveform * loop number

            # Get N waveform sequences from both channels 
            ref_data, chip_data = ss.extract_waves_multi_seq(c, 
                                                        N=N,
                                                        num_samples=num_samples, 
                                                        ref_channel="C1", ref_edge_slope="POS", ref_thresh=ref_thresh,
                                                        chip_channel="C2", chip_edge_slope="NEG", chip_thresh=chip_thresh)
            print(f"\tData acquired")
            # Add approprite offset to time data
            ref_data[0] = ref_data[0] + time_this_loop
            chip_data[0] = chip_data[0] + time_this_loop

            # See if data works for calculating edge offsets
            try:
                # Get time offset btwn falling edges in both channels
                offset_vals = ss.get_offsets(ref_data,
                                    chip_data,
                                    ref_threshold=ref_thresh,
                                    chip_threshold=chip_thresh,
                                    clip=clip)
                
                print(f"\tOffsets calculated")
                
                # Save wave data to files specific to this loop
                np.save(ref_data_file_i, ref_data.T)
                np.save(chip_data_file_i, chip_data.T)
                
                # Just so we don't overflow memory; just for testing
                # os.remove(ref_data_file_i)
                # os.remove(chip_data_file_i)

                print("\tWaveforms saved")

                # Save offset data to file specific to this loop
                np.savetxt(offset_file_i, offset_vals)
                print(f"\tOffsets saved")
            
                del ref_data, chip_data, offset_vals
                loop += 1
            
            # If data isn't properly formatted, discard it and move on to the next loop
            except ValueError:
                print(f"\tDiscarding problematic waveforms from this loop")
            
                del ref_data, chip_data
            
            gc.collect()

        # Combine all offset data into one large file
        with open(combined_offset_file, "w", encoding="utf-8") as outfile:
            for filename in os.listdir(save_dir_offset):
                if filename.endswith(".txt"):
                    filepath = os.path.join(save_dir_offset, filename)

                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                    
                    os.remove(filepath)
        
        # Remove the now empty offset values directory
        os.rmdir(save_dir_offset)
        
        offset_vals_all = np.loadtxt(combined_offset_file)


        mean_val, std_val = ss.calculate_mean_and_std(offset_vals_all)
        
        print(f"\nAverage offset btwn edges: {mean_val}")
        print(f"Stdv of offset time: {std_val}")

        hist, bin_edges = ss.make_histogram_and_gaussian(offset_vals_all, hist_bins=40)
