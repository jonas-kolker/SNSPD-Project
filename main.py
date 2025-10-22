
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
        "DCcompensate": range(0, 8, 1),
        "DFBamp": range(1, 16, 4),
        "DSNSPD": range(0, 128, 16),
        "DAQSW": range(0, 128, 16),
        "VRL": range(0, 32, 4),
        "Dbias_NMOS": range(0, 256, 32),
        "DBias_internal": [0, 1],
        "Dbias_fb_amp": range(0, 128, 16),
        "Dbias_comp": range(0, 128, 16),
        "Dbias_PMOS": range(0, 201, 20),
        "Dbias_ampNMOS": range(0, 128, 16),
        "Ddelay": range(0, 128, 16),
        "Dcomp": range(0, 16, 4),
        "Analoga": ['None', 'Vref', 'Vamp', 'Vcomp'],
        "Dbias_ampPMOS": range(0, 128, 16),
        "DCL": range(0, 16, 4),
        "Dbias_ampn1": range(0, 128, 16),
        "Dbias_ampn2": range(0, 128,16)
    }
    return ranges.get(param_name, [0])

save_dir = "C:\\LeCroy\\ScopeData"

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
              delete_prev_data = True,
              std_cutoff=5, deskew_time=30e-9):   
    
    """
    Acquires waveform data from the scope, triggered by the rising edge of a reference signal followed by the falling edge of a second (chip) signal.
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

        delet_prev_data (bool): Whether or not to delete old folders with the same names as the ones we'll be using (best to keep True)
        std_cutoff (int): Any delay data more than this many stdvs from the mean will be discarded and not considered. Removes extreme outlier data.
        deskew_time (int): Delay ref data by this much in acquisition. Helps align edges in both channels so less data to be collected. 
    
    Returns:
        offset_stdv (float): The standard deviation of the offsets. If std_cutoff isn't 0, this will be the filtered data value
    
    """
    
    # Make appropriate (sub)directories for storing data from each loop
    save_dir = "C:\\LeCroy\\ScopeData"
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

    with MAUI() as c:
        loop = 0
        
        while loop < num_loops:
            print(f"Loop {loop}")
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
            ref_data, chip_data = ss.extract_waves_multi_seq(c, 
                                                        N=N,
                                                        num_samples=num_samples, 
                                                        ref_channel="C1", ref_edge_slope="POS", ref_thresh=ref_thresh,
                                                        chip_channel="C2", chip_edge_slope="NEG", chip_thresh=chip_thresh,
                                                        hold_time=hold_time, deskew_val=deskew_time)
            print(f"\tData acquired")
            
            # Add approprite offset to time data
            ref_data[0] = ref_data[0] + time_this_loop
            chip_data[0] = chip_data[0] + time_this_loop

            # When we get a sequence of waveforms, the first one always seems to have some weird voltage spike. We specify here that all the samples from this first acquisition should be disregarded
            clip = num_samples//3

            # See if data works for calculating edge offsets
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
                
                # So we don't overflow memory; just for testing
                # os.remove(ref_data_file_i)
                # os.remove(chip_data_file_i)
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
        os.rmdir(save_dir_offset)
        
        offset_vals_all = np.loadtxt(combined_offset_file)

        mean_val, std_val = ss.calculate_mean_and_std(offset_vals_all)
        
        # print(f"\nAverage offset btwn edges: {mean_val}")
        # print(f"Stdv of offset time: {std_val}")

        fig, hist, bin_edges, stdv_val = ss.make_histogram_and_gaussian(offset_vals_all, 
                                                                         hist_bins=40, 
                                                                         stdv_cutoff=std_cutoff,
                                                                         return_stdv=True
                                                                         )

        save_path = os.path.join(save_dir, f"hist_{param_name}{sweep_val}.png")

        fig.savefig(save_path)
        plt.close(fig)

        return stdv_val

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
    num_samples = int(1e3) # Number of samples per acquisition segment in the sequence
    N = 1000 # Number of acquisitions per sequence
    num_loops = 1 # Number of sequences 
    
    # div_time = 5e-9 # There are 10 divisons per acquisition
    hold_time = 100e-9 # Chip falling edge must occur within this many seconds after ref rising edge to trigger acq
    deskew_time = 30e-9 # Delay the ref signal by this much, helps align edges btwn channels for data acq purposes
    
    ref_thresh = .05 # Voltage thresholds for reference and chip signals
    chip_thresh = 0.5

    jitter_list = []
    param_val_list = []

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

                    snspd.set_register(**registers)
                    snspd.TX_reg()
                    print(f"\nSet {param} = {val}")

                    stdv_val = scope_acq(param, sweep_val=val,
                                        num_samples=num_samples, N=N, num_loops=num_loops,
                                        div_time=div_time, hold_time=hold_time, deskew_time=deskew_time, 
                                        ref_thresh=ref_thresh, chip_thresh=chip_thresh)
                    
                    param_val_list.append(val)
                    jitter_list.append(stdv_val)

        print("\nSweep completed successfully!")

    plt.plot(param_val_list, jitter_list)
    plt.xlabel("DCcompensate vals")
    plt.ylabel("Delay Stdv")
    plt.title("Delay Standard Deviation (Jitter) vs DCcompensate Params")
    plt.show()