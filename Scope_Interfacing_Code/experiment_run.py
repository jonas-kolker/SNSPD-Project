import json
import os
import scope_script_MDP as ss
from MAUI import MAUI

def load_settings(filename):
    with open(filename, 'r') as f:
        return json.load(f)

if __name__=="__main__":
    
    scope_params = load_settings("experiment_scope_config.json")

    acquisition_mode = scope_params.get("acquisition_mode") # What kinds of acquisitions to get ('single' or 'multi')
    num_acquisitions = scope_params.get("num_acquisitions") # For multi-acquisition mode, how many acquisitions to get
    ref_threshold = scope_params.get("voltage_threshold") # What is the threshold voltage below which to trigger an acquisition
    trigger_channel = scope_params.get("trigger_channel") # Which channel is the trigger/reference signal

    with MAUI() as scope:
        
        # Need to add an additional loop that iterates over chip parameters and interfaces with arduino accordingly

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
            
        
        
        

