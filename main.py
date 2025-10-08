
"""
Main experiment sweep file
Combines SNSPD parameter sweeping with oscilloscope data acquisition.
"""

import json
import time
import numpy as np
from Arduino_SNSPD.classes.Snspd_V2_TEST import Snspd
from Scope_Interfacing_Code.MAUI import MAUI
import Scope_Interfacing_Code.scope_script_MDP as ss


def load_settings(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
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



if __name__ == "__main__":

    #arduino and scope set up
    arduino_port, smu_addr = load_com_ports("Arduino_SNSPD/COM_ports.txt")
    scope_params = load_settings("experiment_scope_config.json")

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

    #need to set these values ask jonas
    acquisition_mode = scope_params.get("acquisition_mode")
    num_acquisitions = scope_params.get("num_acquisitions")
    ref_threshold = scope_params.get("voltage_threshold")
    trigger_channel = scope_params.get("trigger_channel")

    with Snspd(arduino_port) as snspd, MAUI() as scope:
        print("\nStarting parameter sweep")

        for param in parameters.keys():
            print(f"Sweeping parameter: {param}")

            for val in sweep_values(param):
                registers = parameters.copy()
                registers[param] = val

                snspd.set_register(**registers)
                snspd.TX_reg()
                print(f"Set {param} = {val}")

                if acquisition_mode == 'single':
                # Get single acquisition data
                    ref_wave, chip_wave = ss.extract_waves_once(scope=scope,
                                                        ref_thresh=ref_threshold,
                                                        trigger_channel=trigger_channel)
        
                elif acquisition_mode == "multi":
                # Get a list of the data from multiple acquisitions
                    ref_waves_list, chip_waves_list = ss.extract_waves_multi(scope = scope, 
                                                ref_thresh = ref_threshold, 
                                                N = num_acquisitions,
                                                trig_channel=trigger_channel)
                    
                    print(f"Acquired {num_acquisitions} traces for {param}={val}")

                #delay for a bit
                time.sleep(1)

        print("\nSweep completed successfully!")
