
"""
Main experiment sweep file
Combines SNSPD parameter sweeping with oscilloscope data acquisition.
"""

import json
import win32com.client
import time
import numpy as np
from Arduino_SNSPD.classes.Snspd_V2_TEST import Snspd
from Scope_Interfacing_Code.MAUI import MAUI
import Scope_Interfacing_Code.scope_stuff_MDP as ss
import shutil
import matplotlib.pyplot as plt
import os, gc
    
def load_com_ports(filename):
    """
    Reads a simple text file containing connection information (e.g., COM ports,
    instrument addresses) and returns each non-empty line as an entry.

    Parameters:
        filename (str): Path to a plain-text file. Each line should contain one
            value (e.g., the Arduino COM port on the first line and the SMU
            address on the second line).

    Returns:
        list[str]: A list of strings corresponding to each line in the file,
        in order.

    Notes:
        - The function does no validation or stripping beyond splitting lines.
        - Will raise FileNotFoundError if the file does not exist.
    """    
    with open(filename, 'r') as f:
        return f.read().splitlines()

# Ranges for parameters(snspd registers)
def sweep_values(param_name):
    """
    Maps a given SNSPD register/parameter name to the iterable of values that
    will be swept during the experiment.

    Parameters:
        param_name (str): The name of the parameter to sweep. Must be one of
            the keys defined in the internal 'ranges' mapping below.

    Returns:
        iterable: A range, list, or other iterable of values to sweep over for
        the specified parameter. If the name is not recognized, returns [0].

    Defined sweeps:
        - DCcompensate
        - DFBamp        
        - DSNSPD        
        - DAQSW         
        - VRL           
        - Dbias_NMOS    
        - DBias_internal
        - Dbias_fb_amp  
        - Dbias_comp    
        - Dbias_PMOS    
        - Dbias_ampNMOS 
        - Ddelay        
        - Dcomp         
        - Analoga       
        - Dbias_ampPMOS 
        - DCL           
        - Dbias_ampn1   
        - Dbias_ampn2   

    Notes:
        - The return type varies by parameter (range vs list), but all are
          iterable and suitable for 'for' loops.
    """

    ranges = {
        "DCcompensate": range(0, 8),
        "DFBamp": range(1, 16),
        "DSNSPD": range(0, 128),
        "DAQSW": range(0, 128),
        "VRL": range(0, 32),
        "Dbias_NMOS": range(0, 256),
        "DBias_internal": [0, 1],
        "Dbias_fb_amp": range(0, 128),
        "Dbias_comp": range(0, 128),
        "Dbias_PMOS": range(0, 201),
        "Dbias_ampNMOS": range(0, 128),
        "Ddelay": range(0, 128),
        "Dcomp": range(0, 16),
        "Analoga": ['None', 'Vref', 'Vamp', 'Vcomp'],
        "Dbias_ampPMOS": range(0, 128),
        "DCL": range(0, 16),
        "Dbias_ampn1": range(0, 128),
        "Dbias_ampn2": range(0, 128)
    }
    return ranges.get(param_name, [0])

def clear_folder(folder_path):
    """
    Deletes all files and folders inside `folder_path`, but leaves the folder itself intact.

    Parameter:
        folder_path (str or pathlib.Path): Folder to be cleared
    """
    for filename in os.listdir(folder_path):
        
        file_path = os.path.join(folder_path, filename)
       
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)          # remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)      # remove directory and all contents
        
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def scope_acq(param_name, sweep_val, 
              num_samples = int(10), N = 3, num_loops = 10, 
              div_time = 50e-9, hold_time = 900e-9,
              ref_channel="C1", chip_channel="C2",
              ref_thresh = .08, chip_thresh = 0.00, 
              std_cutoff=5, deskew_time=30e-9,
              hist_bins = 900):   
    
    """
    Acquires -- and processes -- waveform data from the scope, triggered by the rising edge of a reference signal followed by the falling edge of a second (chip) signal.
    Waveform data is acquired in sequence bursts - the number of sequences given by num_loops. A sequence will contain N waveforms, and each 
    waveform is made up of num_samples number of datapoints. Sequence data will be saved in folders corresponding to the value of the chip parameter 
    of interest at that time.

    Parameters:
        param_name (str): Name of the chip parameter being swept over in the experiment. Will be used for naming folders/files
        sweep_val (int): Current value of the chip parameter being swept over. Will be used for naming folders/files
        
        num_samples (int): The number of datapoints to collect in each individual waveform acquisition
        N (int): The number of waveforms to collect for each sequence pulled from the scope
        num_loops (int): The number of sequences to pull from the scope
        
        div_time (float): Duration of each time division. Each waveform acquisition will span 10 divisions. Each sequence is N*10 divisions
        hold_time (float): Maximum time btwn rising ref edge and falling chip edge for scope to trigger an acquisition
        deskew_time (float): Delay the ref signal by this much, helps align edges btwn channels for data acq purposes
        
        ref_channel, chip_channel (str): Scope channels for ref signal, chip signal
        ref_thresh, chip_thresh (float): Voltage thresholds for edges (rising edge for ref, falling edge for chip) used to trigger events and calculate delays

        std_cutoff (int): Any delay data more than this many stdvs from the mean will be discarded and not considered. Removes extreme outlier data.
        deskew_time (int): Delay ref data by this much in acquisition. Helps align edges in both channels so less data to be collected. 
    
    Returns:
        offset_stdv (float): The fitted standard deviation (jitter) of the offsets. If std_cutoff isn't 0, this will be the filtered data value
        offset_stdv_err (float): The error associated with this fitted value
    """
    
    # Make appropriate subdirectories for storing data from each loop
    # 
    # save_dir = "C:\\LeCroy\\ScopeData"
    save_dir_ref = save_dir + f"\\ReferenceWaveforms_{param_name}{sweep_val}"
    save_dir_chip = save_dir + f"\\ChipWaveforms_{param_name}{sweep_val}"
    # save_hist = save_dir + f"\\Histograms_{param_name}{sweep_val}"
    save_dir_offset = save_dir + f"\\OffsetVals_{param_name}{sweep_val}"

    # Create final files that will hold "all" the data
    combined_offset_file = os.path.join(save_dir, f"offset_values_all_{param_name}{sweep_val}.txt")

    # Delete folders with data files from previous experiments
    # if delete_prev_data:
    #    clear_folder(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_ref, exist_ok=True)
    os.makedirs(save_dir_chip, exist_ok=True)
    os.makedirs(save_dir_offset, exist_ok=True)
    
    clip = N*16 + 32

    with MAUI() as c:
        loop = 0
        
        while loop < num_loops:
            print(f"{param}: {sweep_val}: Loop {loop}")
            # Reset scope settings
            c.reset()

            # Set appropriate voltage scales and time divisions for acquisition
            c.set_vertical_scale(ref_channel, .05) 
            c.set_vertical_scale(chip_channel, .35)
            c.set_timebase(div_time)

            c.idn() # Needed for scope acquisitions to work for some reason

            # Create files to save waveforms from this loop
            ref_data_file_i = os.path.join(save_dir_ref, f"ref_data_{loop:03}.npy")
            chip_data_file_i = os.path.join(save_dir_chip, f"chip_data_{loop:03}.npy")
            offset_file_i = os.path.join(save_dir_offset, f"offset_vals_{param_name}{sweep_val}_{loop:03}.txt")

            # Get the absolute time wrt previous loops so that time data between files is distinct
            time_this_loop = N*10*div_time*loop # Number of waveforms * 10 time divisions per waveform * loop number

            # Get N waveform sequences from both channels 
            ref_data, chip_data, real_num_samples = ss.extract_waves_multi_seq(c, 
                                                        N=N,
                                                        num_samples=num_samples, 
                                                        ref_channel="C1", ref_edge_slope="POS", ref_thresh=ref_thresh,
                                                        chip_channel="C2", chip_edge_slope="NEG", chip_thresh=chip_thresh,
                                                        hold_time=hold_time, deskew_val=deskew_time, clip=clip)
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
                                    mismatch_handling=True,
                                    num_samples=real_num_samples)
                
                print(f"\tOffsets calculated")
                
                # Save wave data to files specific to this loop
                np.save(ref_data_file_i, ref_data)
                np.save(chip_data_file_i, chip_data)
                
                # So we don't overflow memory; just for testing
                os.remove(ref_data_file_i)
                os.remove(chip_data_file_i)
                print("\tWaveforms saved")

                # Save offset data to file specific to this loop
                np.savetxt(offset_file_i, offset_vals)
                print(f"\tOffsets saved")
            
                del ref_data, chip_data, offset_vals
                loop += 1
            
            # If number of ref and chip edges don't match (and mismatch_handling==False), discard the sequence try again
            except ValueError:
                print(f"\tDiscarding problematic waveforms from this loop and retrying")
            
                del ref_data, chip_data
            
            gc.collect()

        # Combine all offset data into one large file and delete individual files
        with open(combined_offset_file, "w", encoding="utf-8") as outfile:
            for filename in os.listdir(save_dir_offset):
                if filename.endswith(".txt"):
                    filepath = os.path.join(save_dir_offset, filename)

                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                    
                    os.remove(filepath)
        
        # Remove the now empty offset values directory        
        offset_vals_all = np.loadtxt(combined_offset_file)
        print(f"\nTotal # of offsets for this measurement: {len(offset_vals_all)}")
        
        # print(f"\nAverage offset btwn edges: {mean_val}")
        # print(f"Stdv of offset time: {std_val}")

        fig, offset_stdv, offset_stdv_err, bin_width = ss.make_histogram_and_gaussian(offset_vals_all, 
                                                                         hist_bins=hist_bins, 
                                                                         stdv_cutoff=std_cutoff)

        save_path = os.path.join(save_dir, f"hist_{param_name}{sweep_val}.png")

        fig.savefig(save_path)
        plt.close(fig)

        return offset_stdv, offset_stdv_err

if __name__ == "__main__":

    #arduino and scope set up
    arduino_port, smu_addr = load_com_ports("Arduino_SNSPD/COM_ports.txt")

    # Default snspd register values
    SNSPD_Dcode = 20
    RAQSW = 40
    Load = 8
    D_code = int(round(SNSPD_Dcode * 5 / 7))

    parameters = dict(
        DCcompensate=2,
        DFBamp=1,
        DSNSPD=SNSPD_Dcode,
        DAQSW=RAQSW,
        VRL=Load,
        Dbias_NMOS=1,
        DBias_internal=True,
        Dbias_fb_amp=1,
        Dbias_comp=1,
        Dbias_PMOS=1,
        Dbias_ampNMOS=5,
        Ddelay=1,
        Dcomp=14,
        Analoga='None',
        Dbias_ampPMOS=5,
        DCL=8,
        Dbias_ampn1=D_code * 2,
        Dbias_ampn2=D_code
    )

    # Set values for scope interactions
    num_samples = int(500) # Number of samples per acquisition segment in the sequence
    # Min possible value is 500 Samples, max is 10 MSamples

    N = 5000 # Cannot be greater than 5000
    num_loops = 10 # Number of sequences 
    
    div_time = 5e-9 # There are 10 divisons per acquisition
    hold_time = 100e-9 # Chip falling edge must occur within this many seconds after ref rising edge to trigger acq
    deskew_time = 30e-9 # Delay the ref signal by this much, helps align edges btwn channels for data acq purposes

    # Voltage thresholds for reference and chip signals
    ref_thresh = .05#.08
    chip_thresh = 0.5#0
    
    # For plotting
    std_cutoff = 3 # Exclude data more than this many raw stdvs from mean
    hist_bins = 100 # How many bins to include in histogram

    jitter_list = []
    jitter_err_list = []
    param_val_list = []

    # Name global variable where everything will be stored
    save_dir = "C:\\LeCroy\\ScopeData"
    
    # Clear all previous data in save_dir
    clear_folder(save_dir)

    with Snspd(arduino_port) as snspd:
        print("\nStarting parameter sweep")

        for param in parameters.keys():
            print(f"Sweeping parameter: {param}")
            per_value_times = []      # seconds for each sweep value

            for val in sweep_values(param):
                if param == "DCcompensate":
                    registers = parameters.copy()
                    registers[param] = val
                    
                    t0 = time.time()
                    snspd.set_register(**registers)
                    set = snspd.TX_reg()

                    if set != True:
                        print(f"{param} not set correctly")
                        break

                    print(f"\nSet {param} = {val}")
                    
                    # Acquire data from scope and calculate jitter
                    stdv_val, stdv_err = scope_acq(param, sweep_val=val,
                                        num_samples=num_samples, N=N, num_loops=num_loops,
                                        div_time=div_time, hold_time=hold_time, deskew_time=deskew_time, 
                                        ref_thresh=ref_thresh, chip_thresh=chip_thresh, 
                                        std_cutoff=std_cutoff)
                    
                    elapsed = time.time() - t0
                    per_value_times.append(elapsed)
                    param_val_list.append(val)
                    
                    # Store jitter values
                    jitter_list.append(stdv_val)
                    jitter_err_list.append(stdv_err)

            # --- per-parameter summary ---
            num_points = len(per_value_times)
            total_time_s = sum(per_value_times) if num_points else 0.0
            mean_time_s = (total_time_s / num_points) if num_points else 0.0

            print(
                f"\nSummary for {param}:\n"
                f"  Sweep points:        {num_points}\n"
                f"  Total time:          {total_time_s:.2f} s ({total_time_s/60:.2f} min)\n"
                f"  Mean per value:      {mean_time_s:.2f} s\n"
            )
        print("\nSweep completed successfully!")

    fwhm_vals = np.asarray(jitter_list) * 2*np.sqrt(2*np.log(2))
    fwhm_errs = np.asarray(jitter_err_list) * 2*np.sqrt(2*np.log(2))

    plt.errorbar(param_val_list, jitter_list, jitter_err_list, fmt="-bo")
    plt.xlabel("VRL vals")
    plt.ylabel("Delay Stdv")
    plt.title("Delay Standard Deviation (Jitter) vs DCcomp")
    plt.show()