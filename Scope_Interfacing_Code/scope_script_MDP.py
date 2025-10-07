import matplotlib.pyplot as plt
import MAUI
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import time
import pandas as pd

# Things to work on still:
#    - Adjust horizontal (time) scale and range for acquisitions
#    - Adjust str_length based on number of points in each acquisition
#    - Something tells me exctract_waves_multi() will cause memory issues if N is too big (how big is that??)
#    - Look into using sequence mode instead of normal mode for multi-trigger acquisitions (should be faster, but too big N may cause more problems than with normal mode)
#    - While acquiring data, dynamically fit a guassian to the histogram and only stop acquisition when the fit converges below a certain error

# - - - - - - - - - - - - - - - - - - - Getting Jitter from Built-In Scope Histogram Function - - - - - - - - - - - - - - - - - - - 

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

# - - - - - - - - - - - - - - - - - - - - - - Getting Jitter from Raw Scope Waveform Data - - - - - - - - - - - - - - - - - - - - - - 

def check_number_of_points(scope, channel):
    """
    Checks and prints the default number of points in the acquisition for the specified channel.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        channel (str): The channel to check (e.g., "C1", "C
    """
    scope.write(f"VBS? 'return = app.Acquisition.{channel}.Out.Result.NumPoints'")
    num_points = int(float(scope.ReadString(1000)))
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


def extract_waves_once(scope, ref_thresh, trig_channel="C1"):
    """
    Retrieves waveforms from both channels of the scope a single time after triggering on a falling edge on channel 1.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        ref_thresh (float): The voltage level at which to trigger.
        trig_channel (str): The channel to set the trigger on (default is "C1").

    Returns:
        ref_data (np.array): Array of reference signal timestamps and amplitudes.
        chip_data (np.array): Array of chip signal timestamps and amplitudes.
    """
        
    # Stop any previous acquisitions and clear buffers
    scope.set_trigger_mode("STOP")
    scope.write("CLEAR")

    # Set the trigger to falling edge on channel 1 below threshold voltage
    set_falling_edge_trigger(scope, trig_channel, ref_thresh)

    # Indicate single acquisition mode
    scope.set_trigger_mode("SINGLE") 

    # Arm acquisition and wait for completion
    scope.trigger()
    scope.wait()

    # Retrieve waveforms from both channels
    time_array_r, ref_array = scope.get_waveform_numpy(channel="C1", str_length=8000) # How big should str length be?
    time_array_c, chip_array = scope.get_waveform_numpy(channel="C2", str_length=8000)

    # Check if time arrays match each other
    if not np.array_equal(time_array_r, time_array_c):
        raise ValueError("Time arrays from both channels do not match.")

    # Combine time and amplitude data into single arrays
    ref_data = np.asarray([time_array_r, ref_array])
    chip_data = np.asarray([time_array_c, chip_array])
    
    return ref_data, chip_data


def extract_waves_multi(scope, ref_thresh, N, trig_channel="C1"):
    """
    Retrieves waveforms from both channels of the scope N times (triggered by falling edge)
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        ref_thresh (float): The voltage level at which to trigger.
        N (int): The number of triggered waveforms from each channel to acquire
        trig_channel (str): The channel to set the trigger on (default is "C1").

    Returns:
        ref_waves_list (list of np.arrays): List of reference signal timestamps and amplitudes arrays.
        chip_waves_list (list of np.arrays): List of chip signal timestamps and amplitudes arrays.
    """

    # Number of triggered acquisitions
    n = 0

    # Lists to hold result arrays
    ref_waves_list = []
    chip_waves_list = []
    
    # Stop any previous acquisitions and clear buffers
    scope.set_trigger_mode("STOP")
    scope.write("CLEAR")

    # Set the trigger to falling edge on channel 1 below threshold voltage
    set_falling_edge_trigger(scope, trig_channel, ref_thresh)
    
    # Indicate normal trigger mode
    scope.set_trigger_mode("NORM")

    # Acquire N triggered waveforms
    while n < N:

        # Wait until acquisition is complete
        scope.wait()

        # Get waveforms from both channels
        time_array_r, ref_array = scope.get_waveform_numpy(channel="C1", str_length=1000) # How big should str length be?
        time_array_c, chip_array = scope.get_waveform_numpy(channel="C2", str_length=1000)

        # Check if time data from both channels match
        if not np.array_equal(time_array_r, time_array_c):
            raise ValueError("Time arrays from both channels do not match.")
        
        # Combine time and amplitude data into single arrays
        ref_data = np.asarray([time_array_r, ref_array])
        chip_data = np.asarray([time_array_c, chip_array])
        
        # Append data from that acquisition to lists
        ref_waves_list.append(ref_data)
        chip_waves_list.append(chip_data)

        n += 1

    scope.set_trigger_mode("STOP")
    return ref_waves_list, chip_waves_list

def extract_waves_multi_seq(scope, ref_thresh, N, num_samples, trig_channel="C1"):
    """
    Retrieves waveforms from both channels of the scope N times (triggered by falling edge). Waveform
    segments are stored on scope until all acquisitions are complete, then they're transferred to the pc.
    
    Parameters:
        scope (MAUI.MAUI): An instance of the MAUI class for scope communication.
        ref_thresh (float): The voltage level at which to trigger.
        N (int): The number of triggered waveforms from each channel to acquire
        num_samples (int): The number of samples to acquire per segment (ie the size of the segment)
        trig_channel (str): The channel to set the trigger on (default is "C1").

    Returns:
        ref_waves_list (list of np.arrays): List of reference signal timestamps and amplitudes arrays.
        chip_waves_list (list of np.arrays): List of chip signal timestamps and amplitudes arrays.
    """

    # Stop any previous acquisitions and clear buffers
    scope.set_trigger_mode("STOP")
    scope.write("CLEAR")

    # Set the trigger to falling edge on channel 1 below threshold voltage
    set_falling_edge_trigger(scope, trig_channel, ref_thresh)

    # Set sequence mode to be on for N segments
    scope.write(F"SEQ ON, {N}, {num_samples}")

    # Set trigger mode to single
    scope.set_trigger_mode("SINGLE")
    scope.trigger()
    scope.wait()

    # Retrieve waveforms from both channels -- SHOULD return all segments together
    time_array_r, ref_array = scope.get_waveform_numpy(channel="C1", str_length=8000) # How big should str length be?
    time_array_c, chip_array = scope.get_waveform_numpy(channel="C2", str_length=8000)

    # Check if time arrays match each other
    if not np.array_equal(time_array_r, time_array_c):
        raise ValueError("Time arrays from both channels do not match.")

    # Combine time and amplitude data into single arrays
    ref_data = np.asarray([time_array_r, ref_array])
    chip_data = np.asarray([time_array_c, chip_array])
    
    return ref_data, chip_data


def get_offsets(ref_data, chip_data):
    """
    Calculates the timing offset between reference and chip falling edge detection signals.
    
    Parameters:
        ref_array (np.array): Array of reference signal data. First axis should be time, second axis signal amplitude.
        chip_array (np.array): Array of chip signal data. First axis should be time, second axis signal amplitude.

    Returns:
        offset_vals (np.array): Array of time differences between falling edge events in chip and reference
    """
    
    # Extract amplitude data
    ref_array = ref_data[1]
    chip_array = chip_data[1]

    # Check if time arrays match
    if not np.array_equal(ref_data[0], chip_data[0]):
        raise ValueError("Time arrays from both channels do not match.")
    
    time_array = np.array(ref_data)[0]  
    
    # Confirm all arrays are of the same length
    min_length = min(len(ref_array), len(chip_array))
    ref_array = ref_array[:min_length]
    chip_array = chip_array[:min_length]

    # Get threshold values for both signals below which we consider detection events
    ref_threshold = (np.max(ref_array) + np.min(ref_array)) / 2
    chip_threshold = (np.max(chip_array) + np.min(chip_array)) / 2

    # Find indices where signals cross the threshold (falling edge detection)
    r_above = ref_array > ref_threshold
    c_above = chip_array > chip_threshold
    ref_crossings_indices  = np.where( (r_above[:-1]) & (~r_above[1:]) )[0]
    chip_crossings_indices = np.where( (c_above[:-1]) & (~c_above[1:]) )[0]

    # Check that each channel has corresponding falling edge events
    if len(ref_crossings_indices) != len(chip_crossings_indices):
        print(f"Number of reference singla threshold crossings: {len(ref_crossings_indices)}")
        print(f"Number of chip threshold crossings: {len(chip_crossings_indices)}")
        raise ValueError("Mismatch in number of detection events between reference and chip signals.")
    
    # Calculate time differences between falling edge events
    ref_crossing_times = time_array[ref_crossings_indices]
    chip_crossing_times = time_array[chip_crossings_indices]
    offset_vals = chip_crossing_times - ref_crossing_times

    return offset_vals


def make_historgram_and_guassian(offset_vals, plot=True, hist_bins=30):
    """
    Create a histogram of the offset values and fit a guassian to it

    Parameters:
        offset_vals (np.array): Array of time differences between falling edge events in chip and reference
        plot (bool): Whether to plot the histogram and fitted guassian
        hist_bins (int): Number of bins to use in the histogram
    
    Returns:
        hist (np.array): Array of histogram bin counts
        bin_edges (np.array): Array of histogram bin edges 
        popt (np.array): Array of the optimal values for the fitted parameters [A, mu, sigma] (sigma is jitter in this case!!)
        err (np.array): Array of the standard errors of the fitted parameters [A_err, mu_err, sigma_err]
        
    """
    def guassian(x, A, mu, sigma):
        return A * norm.pdf(x, mu, sigma)
    
    # Turn offset values into histogram with specified number of bins
    hist, bin_edges = np.histogram(offset_vals, bins=hist_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit a guassian to the histogram data
    popt, pcov = curve_fit(guassian, 
                           bin_centers, 
                           hist, 
                           p0=[np.max(hist), np.mean(offset_vals),  np.std(offset_vals)])
    
  
    err = np.sqrt(np.diag(pcov))

    if plot:
        # Plot histogram and fitted guassian
        plt.figure(figsize=(8,5))
        plt.hist(offset_vals, bins=hist_bins)
        x_fit = np.linspace(min(offset_vals), max(offset_vals), 1000)
        y_fit = guassian(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r--', label= r'$\sigma=$' + f'{popt[2]:.2e} ± {err[2]:.2e}')
        plt.xlabel('Time Offset (s)')
        plt.ylabel('Counts')
        plt.title('Histogram of Time Offsets with Fitted Guassian')
        plt.legend()
        plt.show()

    return hist, bin_edges, popt, err

    
