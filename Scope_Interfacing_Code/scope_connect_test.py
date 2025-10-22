import win32com.client #imports the pywin32 library
import scope_stuff_MDP as ss
from MAUI import MAUI
import matplotlib.pyplot as plt
import shutil
import numpy as np
import os, gc

if __name__=="__main__":
    
    delete_prev_data = True # Delete data from previous experiments
    
    num_samples = int(1e3) # Number of samples per acquisition segment in the sequence
    N = 10 # Number of acquisitions per sequence
    num_loops = 10 # Number of sequences 
    
    div_time = 5e-9 # There are 10 divisons per acquisition
    hold_time = 100e-9 # Chip falling edge must occur within this many seconds after ref rising edge to trigger acq
    deskew_time = 30e-9 # Delay the ref signal by this much, helps align edges btwn channels for data acq purposes

    # Voltage thresholds for reference and chip signals
    ref_thresh = .05#.08
    chip_thresh = 0.5#0

    # Number of initial samples to discard when processing data
    # There's consistently a weird signal spike during the first acquisition, so we discard some data
    clip = num_samples//3

    # Make appropriate (sub)directories for storing data from each loop
    save_dir = "C:\\LeCroy\\ScopeData"
    save_dir_ref = save_dir + "\\ReferenceWaveforms_test"
    save_dir_chip = save_dir + "\\ChipWaveforms_test"
    save_dir_offset = save_dir +"\\OffsetVals_test"

    # Create final files that will hold data from all sequences
    combined_offset_file = os.path.join(save_dir, "offset_values_all_test.txt")
    # combined_ref_data_file = os.path.join(save_dir, "ref_data_net.npy")
    # combined_chip_data_file = os.path.join(save_dir, "chip_data_net.npy")

    if delete_prev_data:
        
        # Delete folders with data files
        if os.path.exists(save_dir_ref):
            shutil.rmtree(save_dir_ref)
        if os.path.exists(save_dir_chip):
            shutil.rmtree(save_dir_chip)
        if os.path.exists(save_dir_offset):
            shutil.rmtree(save_dir_offset)

        # Delete cumulative files
        if os.path.exists(combined_offset_file):
            os.remove(combined_offset_file)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_ref, exist_ok=True)
    os.makedirs(save_dir_chip, exist_ok=True)
    os.makedirs(save_dir_offset, exist_ok=True)

    with MAUI() as c:
        loop = 0
        
        while loop < num_loops:
            print(f"Loop {loop}")
            # Reset scope settings
            c.reset()

            # Set appropriate voltage scales and time divisions for acquisition
            c.set_vertical_scale("C1", .05) 
            c.set_vertical_scale("C2", .35)
            c.set_timebase(div_time)

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
                                                        chip_channel="C2", chip_edge_slope="NEG", chip_thresh=chip_thresh,
                                                        hold_time=hold_time, deskew_val=deskew_time)
            print(f"\tData acquired")
            print(f"Shape of ref_data is {ref_data.shape}")
            
            # Add approprite offset to time data
            ref_data[0] = ref_data[0] + time_this_loop
            chip_data[0] = chip_data[0] + time_this_loop

            try:
                # Get time offset btwn falling edges in both channels
                offset_vals = ss.get_offsets(ref_data,
                                    chip_data,
                                    ref_threshold=ref_thresh,
                                    chip_threshold=chip_thresh,
                                    clip=clip,
                                    mismatch_handling=True,
                                    num_samples=num_samples)
                
                print(f"\tOffsets calculated")
                
                # Save wave data to files specific to this loop
                np.save(ref_data_file_i, ref_data)
                np.save(chip_data_file_i, chip_data)
                print("\tWaveforms saved")
                
                # # So we don't overflow memory; just for testing
                # os.remove(ref_data_file_i)
                # os.remove(chip_data_file_i)

                # Save offset data to file specific to this loop
                np.savetxt(offset_file_i, offset_vals)
                print(f"\tOffsets saved")
            
                del ref_data, chip_data, offset_vals
                loop += 1
            
            # If data isn't properly formatted, discard it and move on to the next loop
            except ValueError as e:
                print(str(e))
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

        fig, hist, bin_edges = ss.make_histogram_and_gaussian(offset_vals_all, hist_bins=100, stdv_cutoff=5, plot=True)
