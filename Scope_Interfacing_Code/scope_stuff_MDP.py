import matplotlib.pyplot as plt
# import MAUI
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import time

# - - - - - - - - - - - - - - - - - - -  Working with Built-In Scope Histogram Function (NOT TESTED) - - - - - - - - - - - - - - - - - - - 

def setup_jitter_histogram(scope, ch1="C1", ch2="C2", polarity1="FALL", polarity2="FALL", measurement_slot="P1"):
    """
    Configure the scope to build a jitter histogram between two channels using the built-in DELAY measurement.

    scope (MAUI.MAUI): An instance of the MAUI class for scope communication
    ch1, ch2 (str): Channel names 
    polarity1, polarity2 (str): "RISE" or "FALL"
    measurement_slot (str): Measurement slot to use on scope
    """

    # Assign DELAY measurement to parameter slot for offset between ch1 and ch2 edges
    scope.write(f"VBS 'app.Measure.{measurement_slot}.ParamEngine = \"Delay\"'")
    scope.write(f"VBS 'app.Measure.{measurement_slot}.Source1 = \"{ch1}\"'")
    scope.write(f"VBS 'app.Measure.{measurement_slot}.Source2 = \"{ch2}\"'")
    scope.write(f"VBS 'app.Measure.{measurement_slot}.Edge1Polarity = \"{polarity1}\"'")
    scope.write(f"VBS 'app.Measure.{measurement_slot}.Edge2Polarity = \"{polarity2}\"'")

    # Enable statistics and histogram view
    scope.write(f"VBS 'app.Measure.Statistics.State = \"ON\"'")
    scope.write(f"VBS 'app.Measure.Statistics.HistogramView = \"ON\"'")

    # Clear previous sweeps
    scope.write("VBS 'app.Measure.ClearSweeps'")

    # (Optional) force histogram to display on screen
    scope.write(f"VBS 'app.Display.Histogram = \"{measurement_slot}\"'")

    print(f"Jitter histogram configured for {ch1}->{ch2} ({polarity1} to {polarity2})")


def extract_histogram_to_csv(scope, filename="C:\\LeCroy\\JitterHist.csv", measurement_slot="P1"):
    """
    Saves the histogram for the specified measurement slot to a CSV file
    and returns a Pandas dataframe.

    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication
        filename (str): Full path to save the CSV file
        measurement_slot (str): Measurement slot to export ("P1", "P2", etc)
    """

    # Export histogram of measurement slot to file
    scope.write(f"VBS 'app.Measure.{measurement_slot}.Histogram.Export \"{filename}\"'")

    # Give the scope a moment to finish writing
    time.sleep(0.5)

    print(f"Histogram exported to {filename}")

    # Load into Python if accessible over shared folder or mapped drive
    try:
        df = pd.read_csv(filename)
        print("Successfully loaded histogram into DataFrame.")
        return df
    except:
        print("Couldn't read back CSV automatically. Ensure the file is accessible.")
        return None

# - - - - - - - - - - - - - - - - - - - - - -  Working with Raw Scope Waveform Data (USED IN MAIN.PY) - - - - - - - - - - - - - - - - - - - - - - 

def check_number_of_points(scope, channel):
    """
    Checks and prints the default number of points in the acquisition for the specified channel.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        channel (str): The channel to check (e.g., "C1", "C
    """
    scope.write(f"VBS? 'return = app.Acquisition.{channel}.Out.Result.NumPoints'")
    num_points = int(float(scope.read(1000)))
    print("Default NumPoints:", num_points)


def set_falling_edge_trigger(scope, channel, ref_thresh):
    """
    Configures the oscilloscope to trigger on a falling edge for the specified channel at the given level.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        channel (str): The channel to set the trigger on (e.g., "C1", "C2").
        ref_thresh (float): The reference voltage level at which to trigger.
    """
    scope.write(r"""VBS 'app.acquisition.trigger.type = "edge" ' """)
    scope.write(f"""VBS 'app.acquisition.trigger.source = "{channel}" ' """)
    scope.write(r"""VBS 'app.acquisition.trigger.edge.slope = "Negative" ' """)
    scope.write(f"""VBS 'app.acquisition.trigger.edge.level = "{ref_thresh} V" ' """)


def set_edge_qualified_trigger(scope, ref_channel="C1", ref_edge_slope="POS", ref_thresh=0,
                               chip_channel="C2", chip_edge_slope="NEG", chip_thresh=0, hold_time=50e-9):
    """
    Set an edge qualified trigger btwn two channels. Trigger goes off only if the chip edge is detected, qualified by the reference edge
    before it.

    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        ref_channel (str): Reference channel
        ref_edge_slope (str): Falling vs rising edge for trigger
        ref_thres (float): Threshold voltage
        chip_channel (str): Chip signal channel
        chip_edge_slope (str): Falling vs rising edge for trigger
        chip_thres (float): Threshold voltage
        hold_time (int): Chip falling edge must occur within this many seconds after ref rising edge

    """
    # Set the trigger to be edge qualified with the first source and qualifier sources set. No hold time limit
    scope.write(f"TRSE TEQ,SR,{chip_channel},QL,{ref_channel},HT,TL,HV,{hold_time}")

    # Set the trigger level for the reference and chip signals
    scope.write(f"{ref_channel}:TRLV {ref_thresh}V")
    scope.write(f"{chip_channel}:TRLV {chip_thresh}V")

    # Set trigger slopes for signals
    scope.write(f"{ref_channel}:TRSL {ref_edge_slope}")
    scope.write(f"{chip_channel}:TRSL {chip_edge_slope}")


def extract_waves_once(scope, ref_thresh=.08, chip_thresh=-.9, 
                       ref_channel="C1", chip_channel="C2",
                       ref_edge_slope="POS", chip_edge_slope="NEG",
                       str_length=1e5
                       ):
    """
    Retrieves waveforms from both channels of the scope a single time after triggering on a falling edge on channel 1.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        ref_thresh (float): The voltage level at which to trigger.
        trig_channel (str): The channel to set the trigger on (default is "C1").
        str_length (int): The number of datapoints from each acquisition to return to the PC

    Returns:
        ref_data (np.array): Array of reference signal timestamps and amplitudes.
        chip_data (np.array): Array of chip signal timestamps and amplitudes.
    """
        
    # Stop any previous acquisitions and clear buffers
    scope.set_trigger_mode("STOP")
    scope.write("CLEAR")

    # Set the trigger to falling edge on channel 1 below threshold voltage
    set_edge_qualified_trigger(scope, ref_channel, ref_edge_slope, ref_thresh,
                               chip_channel, chip_edge_slope, chip_thresh)

    # Indicate single acquisition mode
    scope.set_trigger_mode("SINGLE") 

    # Arm acquisition and wait for completion
    scope.trigger()
    scope.wait()

    # Retrieve waveforms from both channels
    time_array_r, ref_array = scope.get_waveform_numpy(channel="C1", str_length=str_length) # How big should str length be?
    time_array_c, chip_array = scope.get_waveform_numpy(channel="C2", str_length=str_length)

    # Check if time arrays match each other
    if not np.array_equal(time_array_r, time_array_c):
        raise ValueError("Time arrays from both channels do not match.")

    # Combine time and amplitude data into single arrays
    ref_data = np.asarray([time_array_r, ref_array])
    chip_data = np.asarray([time_array_c, chip_array])
    
    return ref_data, chip_data


def extract_waves_multi_seq(scope, N, num_samples, 
                            ref_channel="C1", ref_edge_slope="POS", ref_thresh=.08,
                            chip_channel="C2", chip_edge_slope="NEG", chip_thresh=-.9, 
                            hold_time=50e-9):
    """
    Retrieves waveforms from both channels of the scope N times (triggered by falling edge). Waveform
    segments are stored on scope until all acquisitions are complete, then they're transferred to the pc.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        N (int): The number of triggered waveforms from each channel to acquire. Max is 15,000
        num_samples (int): The number of samples to acquire per segment (ie the size of the segment)
        ref_channel (str): Reference channel
        ref_edge_slope (str): Falling vs rising edge for trigger
        ref_thres (float): Threshold voltage
        chip_channel (str): Chip signal channel
        chip_edge_slope (str): Falling vs rising edge for trigger
        chip_thres (float): Threshold voltage
        hold_time (int): Chip falling edge must occur within this many seconds after ref rising edge

    Returns:
        ref_waves_list (list of np.arrays): List of reference signal timestamps and amplitudes arrays.
        chip_waves_list (list of np.arrays): List of chip signal timestamps and amplitudes arrays.
    """

    # Stop any previous acquisitions and clear buffers
    scope.set_trigger_mode("STOP")
    scope.write("CLEAR")

    # Set sequence mode to be on for N segments
    scope.write(F"SEQ ON, {N}, {num_samples}")

    # Set the trigger to falling edge on channel 1 below threshold voltage
    set_edge_qualified_trigger(scope, 
                               ref_channel, ref_edge_slope, ref_thresh,
                               chip_channel, chip_edge_slope, chip_thresh,
                               hold_time)

    # Set trigger mode to single
    scope.set_trigger_mode("SINGLE")
    scope.trigger()
    scope.wait()

    # Retrieve waveforms from both channels -- SHOULD return all segments together
    time_array_r, ref_array = scope.get_waveform_numpy(channel="C1", str_length=N*num_samples) # How big should str length be?
    time_array_c, chip_array = scope.get_waveform_numpy(channel="C2", str_length=N*num_samples)

    # Check if time arrays match each other
    if not np.array_equal(time_array_r, time_array_c):
        raise ValueError("Time arrays from both channels do not match.")

    # Combine time and amplitude data into single arrays
    ref_data = np.asarray([time_array_r, ref_array])
    chip_data = np.asarray([time_array_c, chip_array])
    
    return ref_data, chip_data


def chunk_data(data_array, num_samples):
    """
    Take an array full of data and seperate an list of smaller arrays with a specified number of samples in each

    Parameters:
        data_array (np.array): A 1D array of values
        num_samples (int): How many values in each chunk
    Returns:
        chunks (list of np.arrays): A (j, num_samples) shaped array where j = len(data_array) // num_samples + (len(data_array) % num_samples)
    
    """
    N = len(data_array)
    indices = np.arange(num_samples, N, num_samples)
    chunks = np.array_split(data_array, indices)

    return chunks 


def get_offsets(ref_data, chip_data, ref_threshold, chip_threshold, clip=0, mismatch_handling=False, num_samples=0):
    """
    Calculates the timing offset between reference and chip falling edge detection signals. If a different number of rising and falling 
    edges are detected, it will either throw a ValueError or break the combined sequence of waveforms into acquisition windows and analyze 
    the edges in each individually. Only specific acquisitions with mismatches will be ignored, rather than the whole file. This option allows 
    you to analyze data that would otherwise be discarded, but means processing will take longer.
    
    Parameters:
        ref_array (np.array): Array of reference signal data. First axis should be time, second axis signal amplitude.
        chip_array (np.array): Array of chip signal data. First axis should be time, second axis signal amplitude.
        ref_threshold (float): Threshold value reference signal below which we consider detection events
        chip_threshold (float): Threshold value chip signal below which we consider detection events
        clip (int): Number of initial samples to be ignored 
        mismatch_handling (bool): If different num of edges counted between channels, either throw an error (if False) or take time to iterate over each individual acquisition (if True).
        num_samples (int): The number of samples per acquisition window. Only needed when mismatch_handling == True
    
    Returns:
        offset_vals (np.array): Array of time differences between falling edge events in chip and reference
    """
    
    # Extract amplitude data
    ref_array = ref_data[1][clip:]
    chip_array = chip_data[1][clip:]

    # Check if time arrays match
    if not np.array_equal(ref_data[0], chip_data[0]):
        print("ERROR: Time arrays from both channels do not match.\n")
    
    time_array = np.array(ref_data)[0][clip:]  
    
    # Confirm all arrays are of the same length
    min_length = min(len(ref_array), len(chip_array))
    ref_array = ref_array[:min_length]
    chip_array = chip_array[:min_length]

    # Find indices where signals cross the threshold (rising edge detection for ref)
    r_below = ref_array < ref_threshold
    ref_crossings_indices  = np.where( (r_below[:-1]) & (~r_below[1:]) )[0]
    
    # Find indices where signals cross the threshold (falling edge detection for chip)
    c_above = chip_array > chip_threshold
    chip_crossings_indices = np.where( (c_above[:-1]) & (~c_above[1:]) )[0]

    # Check that each channel has corresponding falling edge events
    print(f"\tNumber of reference threshold crossings: {len(ref_crossings_indices)}")
    print(f"\tNumber of chip threshold crossings: {len(chip_crossings_indices)}")
    
    # - - - - - - - - - -  - - -  If the number of crossings in each channel don't match - - - - - - - - - - - - - - -  - - - -  - 
    if len(ref_crossings_indices) != len(chip_crossings_indices):
        
        if mismatch_handling == False:
            raise ValueError("Mismatch in number of detection events between reference and chip signals.")
        
        elif mismatch_handling == True:
            
            if num_samples == 0:
                print("ERROR: Need to specify the number of samples per segment\n")

            # Confirm that the total number of datapoints is min_length=N*num_samples (where N is number of acquisitions per sequence)
            # print(f"\tTotal # of samples: {min_length}\n\tSamples per acquisition: {num_samples}")
            print(f"\tHandling mismatches segment by segment")
            # assert min_length % num_samples == 0, "Total samples in sequence doesn't divide by num_samples per acquisition"
            # N = min_length/num_samples
            
            # Break the data into chunks corresponding to each individual sequence segment
            ref_waveforms = chunk_data(ref_array, num_samples)
            chip_waveforms = chunk_data(chip_array, num_samples)
            waveform_time_vals = chunk_data(time_array, num_samples)
            offset_vals = []

            # - - - - - - - - -- - - - - - - - WORK IN PROGRESS - - - - - - - - - - - - - - - - - 
            # # Handle all uniformly sized data
            # ref_waveforms_array = np.array(ref_waveforms[:-1])
            # chip_waveforms_array = np.array(chip_waveforms[:-1])
            # waveform_time_vals = np.array(waveform_time_vals[:-1])

            # # Get smaller, final segment data
            # ref_waveforms_last = np.array(ref_waveforms[-1])
            # chip_waveforms_last = np.array(chip_waveforms[-1])
            # waveform_time_vals_last = np.array(waveform_time_vals[-1])

            # ref_below  = ref_waveforms_array < ref_threshold
            # ref_crossing_
            
            #chip_above = chip_waveforms_array < chip_threshold
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  


            # Try to get offset value for each individual acquisition segment. If there's a mismatch or error, discard that segment
            for i in range(len(ref_waveforms)):
                seg_ref_array = ref_waveforms[i]
                seg_chip_array = chip_waveforms[i]
                seg_time_array = waveform_time_vals[i]

                # For each segment, find indices of threshold crossing (rising edge for ref)
                seg_r_below = seg_ref_array < ref_threshold
                seg_ref_crossing_index  = np.where( (seg_r_below[:-1]) & (~seg_r_below[1:]) )[0]

                # For each segment, find indices of threshold crossing (falling edge for chip)
                seg_c_above = seg_chip_array > chip_threshold
                seg_chip_crossing_index = np.where( (seg_c_above[:-1]) & (~seg_c_above[1:]) )[0]

                # There should be the same number of crossings in each channel, and that number should be 1 per individaul segment
                seg_num_ref_crossings = len(seg_ref_crossing_index)
                seg_num_chip_crossings = len(seg_chip_crossing_index)
                if (seg_num_chip_crossings != seg_num_ref_crossings) or (seg_num_chip_crossings != 1) or (seg_num_ref_crossings != 1):
                    pass
                
                # If there's one crossing per channel in this segment, proceed
                else:
                    seg_ref_crossing_time = seg_time_array[seg_ref_crossing_index]
                    seg_chip_crossing_time = seg_time_array[seg_chip_crossing_index]
                    
                    offset_vals.append(seg_chip_crossing_time - seg_ref_crossing_time)
                    # print(seg_ref_crossing_time-seg_chip_crossing_time)
            
            return np.asarray(offset_vals)
    # - - -  - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - 
    
    # If the number of crossings in each channel match for the entire sequence
    else:
        ref_crossing_times = time_array[ref_crossings_indices]
        chip_crossing_times = time_array[chip_crossings_indices]
        offset_vals = chip_crossing_times - ref_crossing_times

        return offset_vals


def make_histogram_and_gaussian(offset_vals, plot=False, hist_bins=30, stdv_cutoff=0):
    """
    Create a histogram of the offset values and fit a gaussian to it

    Parameters:
        offset_vals (np.array): Array of time differences between falling edge events in chip and reference
        plot (bool): Whether to plot the histogram and fitted gaussian
        hist_bins (int): Number of bins to use in the histogram
        stdv_cutoff (int): Filters out data more than some this many sigmas from the mean. Set to 0 for no cutoff.
    
    Returns:
        fig (plt.Figure): Pyplot figure object that can be saved/displayed
        hist (np.array): Array of histogram bin counts
        bin_edges (np.array): Array of histogram bin edges 
        
        
    """
    # Omit outlier data for prettier histogram if cutoff not set to 0
    if stdv_cutoff != 0:
         mask = np.abs(offset_vals-np.mean(offset_vals)) < stdv_cutoff * np.std(offset_vals)
    else:
        mask = np.ones_like(offset_vals) == 1
    
    filtered_vals = offset_vals[mask]
    hist, bin_edges = np.histogram(filtered_vals, bins=hist_bins)
    
    A = np.max(hist)
    mean = np.mean(filtered_vals)
    stdv = np.std(filtered_vals)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gaussian(x, amp, mu, sigma):
        
        # We multiply by bin width to account for the fact that the gaussian here is measuring
        # counts, NOT a probability distribution
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) #* bin_width

    # Create histogram and fitted gaussian
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(filtered_vals, bins=hist_bins)
    # plt.xlim(mean - stdv_cutoff*stdv, mean + stdv_cutoff*stdv_cutoff)
    
    x_fit = np.linspace(min(bin_edges), max(bin_edges), 1000)
    y_fit = gaussian(x_fit, A, mean, stdv)
    ax.plot(x_fit, y_fit, 'r--', label= r'FWHM=' + f'{ 2*np.sqrt(2*np.log(2)) *stdv:.2e}')
    
    ax.set_xlabel('Time Offset (s)')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of Time Offsets')
    ax.legend()
    
    if plot:
        
        plt.show() 

    return fig, hist, bin_edges


def calculate_mean_and_std(offset_value_list):

    offset_value_array = np.array(offset_value_list)
    mean = np.mean(offset_value_array)
    std_dev = np.std(offset_value_array)

    return mean, std_dev
    
