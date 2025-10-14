import win32com.client #imports the pywin32 library
import scope_script_MDP as ss
from MAUI import MAUI
import matplotlib.pyplot as plt
import time
import numpy as np

if __name__=="__main__":

    avg_dur_list = []
    real_num_samples_list = []

    max_samples_list = np.linspace(100, 1e6, num=10)

    with MAUI() as c:
        
        for num_samples in max_samples_list:
            ind_dur_list = []
            for i in range(5):
                c.reset()

                # c.trigger()
                c.set_vertical_scale("C1", 2)
                c.set_vertical_scale("C2", 2)
                c.set_timebase(1e-6)
                # ss.set_falling_edge_trigger(c, "C1", .2)

                N = 100 # Issues when over 100 for some reason when using Arduino signals
        
                start_t = time.time()
                ref_data, chip_data = ss.extract_waves_multi_seq(c, 
                                                            ref_thresh=0,
                                                            N=N,
                                                            num_samples=num_samples, 
                                                            trig_channel="C1")
                rt_data, rv_data = ref_data[0], ref_data[1]
                ct_data, cv_data = chip_data[0], chip_data[1]
                c.idn()

                duration = time.time()-start_t
                ind_dur_list.append(duration)
            
            avg_dur = np.mean(ind_dur_list)
            avg_dur_list.append(avg_dur)

            real_num_samples = int(c.query("""VBS? 'return=app.acquisition.horizontal.maxsamples'"""))
            real_num_samples_list.append(real_num_samples)

    print(f"Pre-defined num_sample values: {max_samples_list}")
    print(f"Actual num_sample values: {real_num_samples_list}")

    plt.plot(real_num_samples_list, avg_dur_list, '-bo')
    plt.ylabel("Seconds")
    plt.xlabel("Samples per sequence")
    plt.title("Avg (n=5) duration to acquire/transfer 100 wave sequences")
    plt.show()

# plt.plot(rt_data, rv_data, label="Reference")
# plt.plot(ct_data, cv_data, label="Chip")
# plt.legend()
# plt.show()

# offset_vals = ss.get_offsets(ref_data, 
#                              chip_data,
#                              ref_threshold=0,
#                              chip_threshold=0,
#                              clip=num_samples)

# hist, bin_edges, popt, err = ss.make_historgram_and_guassian(offset_vals)


# scope=win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") #creates instance of the ActiveDSO control
# scope.MakeConnection("IP:169.254.250.104") #Connects to the oscilloscope. Substitute your IP address
# scope.WriteString("C1:VDIV .04",1) #Remote Command to set C1 volt/div setting to 20 mV.
# scope.Disconnect() #Disconnects from the oscilloscope