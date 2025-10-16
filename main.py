
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
import Scope_Interfacing_Code.scope_script_MDP as ss
import matplotlib.pyplot as plt
import os, gc
    
def load_com_ports(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

#Ranges for parameters(snspd registers)
def sweep_values(param_name):
    ranges = {
        "DCcompensate": range(0, 8),
        "DFBamp": range(0, 16),
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

def scope_acq(param_name, sweep_val,N = 3,num_samples = int(1e4), div_time = 50e-9, num_loops = 2, ref_thresh = .08, chip_thresh = 0.00):   
    
    clip = num_samples
    
    # Make appropriate (sub)directories for storing data from each loop
    save_dir = "C:\\LeCroy\\ScopeData"
    save_dir_ref = save_dir + f"\\ReferenceWaveforms{param_name}{sweep_val}"
    save_dir_chip = save_dir + f"\\ChipWaveforms{param_name}{sweep_val}"
    save_dir_offset = save_dir + f"\\OffsetVals{param_name}{sweep_val}"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_ref, exist_ok=True)
    os.makedirs(save_dir_chip, exist_ok=True)
    os.makedirs(save_dir_offset, exist_ok=True)

    # Create final files that will hold "all" the data
    combined_offset_file = os.path.join(save_dir, f"offset_values_all{param_name}{sweep_val}.txt")
    combined_ref_data_file = os.path.join(save_dir, f"ref_data_net{param_name}{sweep_val}.npy")
    combined_chip_data_file = os.path.join(save_dir, f"chip_data_net{param_name}{sweep_val}.npy")
    
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

        # hist, bin_edges = ss.make_histogram_and_gaussian(offset_vals_all, hist_bins=40)

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

    with Snspd(arduino_port) as snspd:
        print("\nStarting parameter sweep")

        for param in parameters.keys():
            print(f"Sweeping parameter: {param}")

            for val in sweep_values(param):
                registers = parameters.copy()
                registers[param] = val

                snspd.set_register(**registers)
                snspd.TX_reg()
                print(f"Set {param} = {val}")

                scope_acq(param, val)

        print("\nSweep completed successfully!")
