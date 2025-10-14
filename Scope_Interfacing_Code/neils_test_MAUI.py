# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:11:30 2024

@author: nielsfakkel
"""

from Lulo_Seqc import Sequencer
import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
from parameters import parameters  # The parameter files m2.nodes.parameters
import pyvisa as visa
from FSW43 import FSW43
from MAUI import MAUI
from SMA100B import SMA100B
from SUP_DP832 import SUP_DP832
import os
import shutil

datapath = "N:/Data/LR/"
save_data = True
plot_data = False
if not save_data:
    print("We will not be saving data!!!")

import sys
from sys import exit
def calculate_snr(x_freq, y_pow_dbm, fundamental_freq):
    idx_fund = np.argmin(np.abs(x_freq - fundamental_freq))
    signal_power_dbm = y_pow_dbm[idx_fund]

    # Exclude the fundamental tone
    noise_power_dbm = 10 * np.log10(
        np.sum(10 ** (np.delete(y_pow_dbm, idx_fund) / 10))
    )

    snr = signal_power_dbm - noise_power_dbm
    return snr

def calculate_sfdr(x_freq, y_pow_dbm, fundamental_freq):
    main_tone_idx = np.argmin(np.abs(x_freq - fundamental_freq))
    main_tone_dbm = y_pow_dbm[main_tone_idx]

    # Remove main tone
    y_pow_dbm_no_main = np.copy(y_pow_dbm)
    y_pow_dbm_no_main[main_tone_idx] = -np.inf

    # Find highest spur
    max_spur_dbm = np.max(y_pow_dbm_no_main)

    sfdr = main_tone_dbm - max_spur_dbm
    return sfdr

def calculate_pow(x_freq, y_pow_dbm, fundamental_freq, max_harmonic=7):
    idx_fund = np.argmin(np.abs(x_freq - fundamental_freq))
    signal_power_dbm = y_pow_dbm[idx_fund]
    return signal_power_dbm

def calculate_sinad(x_freq, y_pow_dbm, fundamental_freq, max_harmonic=7):
    idx_fund = np.argmin(np.abs(x_freq - fundamental_freq))
    signal_power_dbm = y_pow_dbm[idx_fund]

    # Find harmonic indices (2nd to Nth)
    num_bins = len(y_pow_dbm)
    harmonic_indices = []
    for h in range(2, max_harmonic + 1):
        harmonic_freq = fundamental_freq * h
        if harmonic_freq > x_freq[-1]:
            break
        idx = np.argmin(np.abs(x_freq - harmonic_freq))
        if 0 <= idx < num_bins:
            harmonic_indices.append(idx)

    # Exclude fundamental + harmonics for noise calculation
    exclude_indices = [idx_fund] + harmonic_indices
    all_indices = np.arange(num_bins)
    noise_indices = np.setdiff1d(all_indices, exclude_indices)

    # Calculate total distortion + noise power in linear domain
    noise_power_mw = np.sum(10 ** (y_pow_dbm[noise_indices] / 10))
    distortion_power_mw = np.sum(10 ** (y_pow_dbm[harmonic_indices] / 10))
    total_dist_noise_mw = noise_power_mw + distortion_power_mw

    sinad = signal_power_dbm - 10 * np.log10(total_dist_noise_mw)
    return sinad

def phdiffmeasure(sig_ref, sig_meas, fs, method='fft', freq=None):
    """
    Measure phase difference, time delay, and amplitude of the dominant tone between two signals.

    Parameters
    ----------
    sig_ref : array_like
        Reference signal.
    sig_meas : array_like
        Measured signal (same length as sig_ref).
    fs : float
        Sampling frequency in Hz.
    method : {'fft', 'xcorr'}, optional
        Method for measuring phase/time difference. Default is 'fft'.
    freq : float or None, optional
        Frequency of interest in Hz. If None, dominant tone is used.

    Returns
    -------
    phase_diff : float
        Phase difference in radians (meas minus ref).
    time_delay : float
        Time difference in seconds (meas delayed relative to ref).
    f0 : float
        Frequency (Hz) at which the phase difference was measured.
    amp : float
        Amplitude of the tone (from the reference signal).
    """
    x = np.asarray(sig_ref)
    y = np.asarray(sig_meas)
    if x.shape != y.shape:
        raise ValueError("sig_ref and sig_meas must be the same length")
    n = x.size

    if method.lower() == 'fft':
        # FFTs
        X = np.fft.fft(x)
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(n, d=1/fs)

        # Choose frequency bin
        if freq is None:
            half = n // 2
            idx = np.argmax(np.abs(X[1:half])) + 1  # skip DC
        else:
            idx = np.argmin(np.abs(freqs - freq))
        f0 = abs(freqs[idx])

        # Phase and amplitude
        phi_x = np.angle(X[idx])
        phi_y = np.angle(Y[idx])
        #phase_diff = phi_y - phi_x
        phase_diff = phi_x - phi_y
        time_delay = phase_diff / (2*np.pi*f0)

        # Amplitude scaling: account for symmetry (real signal)
        if idx == 0 or (n % 2 == 0 and idx == n//2):  # DC or Nyquist
            amp = np.abs(X[idx]) / n
        else:
            amp = 2 * np.abs(X[idx]) / n

        return phase_diff, time_delay, f0, amp

    elif method.lower() == 'xcorr':
        corr = np.correlate(y, x, mode='full')
        lags = np.arange(-n + 1, n)
        lag = lags[np.argmax(corr)]
        time_delay = lag / fs

        # Estimate frequency if not given
        if freq is None:
            X = np.fft.fft(x)
            freqs = np.fft.fftfreq(n, d=1/fs)
            idxf = np.argmax(np.abs(X[1:n//2])) + 1
            f0 = abs(freqs[idxf])
        else:
            f0 = freq

        phase_diff = -2 * np.pi * f0 * time_delay

        # Estimate amplitude from FFT of reference signal
        if freq is None:
            idx = idxf
        else:
            idx = np.argmin(np.abs(freqs - f0))
        if idx == 0 or (n % 2 == 0 and idx == n//2):
            amp = np.abs(X[idx]) / n
        else:
            amp = 2 * np.abs(X[idx]) / n

        return phase_diff, time_delay, f0, amp

    else:
        raise ValueError("method must be 'fft' or 'xcorr'")


# Test gate sequence
parameters.load()
with Sequencer(com='COM5',parameters=parameters) as lr_seqc, \
        SUP_DP832("USB0::0x1AB1::0x0E11::DP8B175201153::0::INSTR") as sup, \
        SMA100B("TCPIP0::169.254.91.32::hislip0::INSTR") as sma, \
        MAUI() as scope:

    # sup.reset()
    sup.set_vol(ch=1, val=5)
    sup.set_vol(ch=2, val=5)
    sup.set_vol(ch=3, val=0.85)

    sup.enable_cur_lim(ch=1, lim=400e-3)
    sup.enable_cur_lim(ch=2, lim=200e-3)
    sup.enable_cur_lim(ch=3, lim=150e-3)

    sup.enable_output(ch=1)
    sup.enable_output(ch=2)
    sup.enable_output(ch=3)

    time.sleep(.5)

    meas_name = "test_mv_cw_am_pm"
    if save_data:
        # Make datastore folder
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S_")
        timedate = time.strftime("%Y%m%d")
        if save_data:
            dirpath = timedate + "/" + timestr + meas_name
            os.makedirs(datapath + timedate, exist_ok=True)
            os.makedirs(datapath + dirpath, exist_ok=True)

            shutil.copy(__file__, datapath + dirpath + "/runfile.py")

    lr_seqc.eop_duration = {  # in samples
        "Gxpi": 300,
        "Gypi": 30,
        "Gxpi2": 20,
        "Gypi2": 20}

    lr_seqc.eop_start_addr = {
        "Gxpi": 0,
        "Gypi": 300,
        "Gxpi2": 600,
        "Gypi2": 800}

    lr_seqc.eop_max_amp = {  # in samples
        "Gxpi": 255,
        "Gypi": 255,
        "Gxpi2": 255,
        "Gypi2": 255}

    F_mw_clk = 5e9
    lr_seqc.F_mw_nco = F_mw_clk/8
    lr_seqc.F0_eop = 0 #15e6
    F_fund = F_mw_clk/2-2*lr_seqc.F0_eop
    lr_seqc.ns_single_tone = 1 #1 # use carrier for AM measurements
    lr_seqc.en_pm_lut = False
    lr_seqc.en_lvl_shift = 1
    ### set the AM code sweep start and end as variables , melbadry
    AM_start = 2
    AM_end = 255

    # initialize SMA100B
    sma.set_power(17.5)
    sma.set_frequency(F_mw_clk)
    sma.set_clk_output('DSI')
    sma.set_clk_frequency(F_fund)
    sma.set_clk_power(0)
    sma.set_status(1)
    sma.set_clk_status(1)

    # initialize cmos
    lr_seqc.flipByte = True
    lr_seqc.reset_LR_reg()
    lr_seqc.clear_arduino_reg()
    lr_seqc.en_controller(high=False)
    lr_seqc.test_tx_ard()
    time.sleep(0.01)



    # Program MW SRAM
    # lr_seqc.compile_sram_electron(env="TwoTone")
    # # lr_seqc.compile_sram_nuclear(env="Gauss")
    # lr_seqc.plot_sram()
    # lr_seqc.program_sram_MW(lr_seqc.electron_SRAM_data, debug=False)
    # lr_seqc.program_sram_RF(lr_seqc.nuclear_SRAM_data, lr_seqc.nuclear_SRAM_addr, debug=False)
    # gst_string_list = [[[["Gxpi:0", "Gypi:0", "Gypi2:0", "Gxpi2:0"], 2]]]

    # # Perform amplitude sweep
    # for amplitude in range(255):
    y = np.arange(AM_start,AM_end,6)
    #y_lin = y
    # y_lin = y + (0.000007289 * y * y) * 255
    #y_lin = 0.001227*y**2 + 0.6873*y # This cal seems ok works
    # y_lin = y + (1.4316E-07 * y * y) * 255
    #sweep_range = lr_seqc.quantize_env(y_lin / np.max(y_lin)).tolist()



    # sweep_range = range(255)
    # sweep_range = [10, 30, 53, 87, 121, 143, 180]
    sweep_range = y.astype(int)

    Curr_VDDmw   = np.zeros([len(sweep_range), 1])
    time_AM = np.zeros([len(sweep_range), 1])
    time_PM  = np.zeros([len(sweep_range), 1])
    #index =  0
    for index, amplitude in enumerate(sweep_range):
    #for amplitude in sweep_range:
        # sma.set_clk_frequency(F_fund)
        # time.sleep(.1)
        # lr_seqc.delay_q_clk = amplitude*19
        # lr_seqc.vb_dac = amplitude #40
        #lr_seqc.delay_q_clk = ((2**8) * (16*8 + 8)) + 16*2 + 2 # + 11*16+12 #9*16 + 8 # 1024*32 + 32*amplitude
        #lr_seqc.delay_i_clk = 0
        lr_seqc.am_clk_sel = 3 #3 # 0-1 Fmw/16 2-3 Fmw/8 4-7 Fmw/32
        lr_seqc.pm_clk_sel = 3 #3
       #lr_seqc.pm_del_sel = 6
        lr_seqc.vbna_dac = 48
        lr_seqc.vbpa_dac = 63 - 48
        lr_seqc.vbnb_dac = 63 - 48
        lr_seqc.vbpb_dac = 48
        cw_amplitude = int(amplitude) #126 #amplitude
        ret_s = lr_seqc.cmos_generate_cw(cw_amplitude)
        ret = lr_seqc.tx_reg(debug=False)
        print("Write successfull?: ", bytes([0xFF]) == ret)
        lr_seqc.rst_and_en_controller()
        lr_seqc.en_controller(high=True)
        time.sleep(.3)

        current = sup.cmd(":MEASure:CURRent? CH3")
        #Curr_VDDmw[index] = current[0]
        print(index)
        Curr_VDDmw[index,0] = current[0]

        print(scope.idn())
        scope.set_trigger_mode("SINGLE")
        scope.trigger()
        scope.wait()
        c1_t, c1_v = scope.get_waveform_numpy(channel = "C1", str_length=8000)  ## interchanged melbadry
        c2_t, c2_v = scope.get_waveform_numpy(channel = "C2", str_length=8000)   ## interchanged melbadry

        phi, td, f_est, amp = phdiffmeasure(c1_v, c2_v, 40e9, method='fft',freq = F_fund)
        print(f"DFT → phase diff = {phi:.3f} rad, time delay = {td:.4f} s at {f_est} Hz, amplitude = {amp}")
        time_PM[index] = np.rad2deg(phi)
        time_AM[index] = amp

        #index = index+1
        # Save the spectra of both inband and out of band
        if save_data:
            channel1 = np.asarray([c1_t, c1_v])
            channel1 = channel1.transpose()
            np.savetxt(datapath + dirpath + "/channel_1_" + str(amplitude) + ".csv", channel1, delimiter=";")

            channel2 = np.asarray([c2_t, c2_v])
            channel2 = channel2.transpose()
            np.savetxt(datapath + dirpath + "/channel_2_" + str(amplitude) + ".csv", channel2, delimiter=";")

            # outband = np.asarray([x_fsw_outband, y_fsw_outband])
            # outband = outband.transpose()
            # np.savetxt(datapath + dirpath + "/spectra_CW_outband_" + str(amplitude) + ".csv", outband, delimiter=";")


        if plot_data:
            plt.figure()
            plt.plot(c1_t, c1_v, color='blue')
            plt.plot(c2_t, c2_v, color='red')
            plt.xlabel("Time [s]")
            plt.ylabel("Signal [V]")
            plt.show()
    #print(amplitude)
    #print(time_AM)
    # ### Start of remeasuring with predistortion ### melbadry
    #
    # #### do the predistortion and put the data in a LUT to load easily
    #
    # ## first need to esimate the ideal curve
    #
    # start_AM = 0
    # end_AM = 0
    # step_size_AM = 0
    # ideal_AM = np.zeros([len(sweep_range), 1])
    # ideal_AM_x = np.zeros([len(sweep_range), 1])
    # ## calculate the start, end and step size of the curve
    # start_AM = time_AM[0]
    # end_AM = time_AM[-1]
    # # print("start and end values")
    # # print(time_AM[0])
    # # print(time_AM[-1])
    # step_size_AM = (end_AM - start_AM) / len(time_AM)
    # c = 0
    # c = start_AM -  step_size_AM*AM_start
    # # print(c)
    # # print(step_size_AM)
    # ## generate the ideal curve
    # for i in range(0, len(time_AM)):
    #     ideal_AM[i] = (i + y[0]) * step_size_AM+c
    #     ideal_AM_x[i] = (i + y[0])
    # # plt.figure()
    # # plt.plot(ideal_AM_x,ideal_AM)
    #
    # ## Generate the lut
    # ## let's say code 51 the ideal value is 50mA and what we get is 70mA
    # ## loop on the undistorted curve and measure the difference between y(n) - y(51)
    # ## then the lut should choose the code with the least error
    # error = np.zeros([len(sweep_range), 1])
    # AM_linearized = np.zeros([len(sweep_range), 1])
    # min_error_index = np.zeros([len(sweep_range), 1])
    #
    # for i in range(0, len(time_AM)):
    #
    #     for k in range(0, len(time_AM)):
    #         error[k] = abs(time_AM[k] - ideal_AM[i])
    #     print(error)
    #     min_error_index = int(np.argmin(error))
    #     AM_linearized[i] = y[min_error_index]
    # y = np.arange(AM_start, AM_end)
    # y_lin = AM_linearized
    #
    # # y_lin = y + (0.000007289 * y * y) * 255
    # # y_lin = 0.001227*y**2 + 0.6873*y # This cal seems ok works
    # # y_lin = y + (1.4316E-07 * y * y) * 255
    # #sweep_range_lin = lr_seqc.quantize_env(y_lin / np.max(y_lin)).tolist()
    # sweep_range_lin = y_lin.astype(int)
    #
    # # sweep_range = range(255)
    # # sweep_range = [10, 30, 53, 87, 121, 143, 180]
    #
    # Curr_VDDmw_lin = np.zeros([len(sweep_range), 1])
    # time_AM_lin = np.zeros([len(sweep_range), 1])
    # time_PM_lin = np.zeros([len(sweep_range), 1])
    #
    # for index, amplitude in enumerate(sweep_range_lin):
    #     # sma.set_clk_frequency(F_fund)
    #     # time.sleep(.1)
    #     # lr_seqc.delay_q_clk = amplitude*19
    #     # lr_seqc.vb_dac = amplitude #40
    #     # lr_seqc.delay_q_clk = ((2**8) * (16*8 + 8)) + 16*2 + 2 # + 11*16+12 #9*16 + 8 # 1024*32 + 32*amplitude
    #     # lr_seqc.delay_i_clk = 0
    #     lr_seqc.am_clk_sel = 3  # 3 # 0-1 Fmw/16 2-3 Fmw/8 4-7 Fmw/32
    #     lr_seqc.pm_clk_sel = 3  # 3
    #     # lr_seqc.pm_del_sel = 6
    #     lr_seqc.vbna_dac = 48
    #     lr_seqc.vbpa_dac = 63 - 48
    #     lr_seqc.vbnb_dac = 63 - 48
    #     lr_seqc.vbpb_dac = 48
    #     cw_amplitude = int(amplitude.item())  # 126 #amplitude
    #     ret_s = lr_seqc.cmos_generate_cw(cw_amplitude)
    #     ret = lr_seqc.tx_reg(debug=False)
    #     print("Write successfull?: ", bytes([0xFF]) == ret)
    #     lr_seqc.rst_and_en_controller()
    #     lr_seqc.en_controller(high=True)
    #     time.sleep(.3)
    #
    #     current = sup.cmd(":MEASure:CURRent? CH3")
    #     Curr_VDDmw_lin[index] = current[0]
    #
    #     print(scope.idn())
    #     scope.set_trigger_mode("SINGLE")
    #     scope.trigger()
    #     scope.wait()
    #     c1_t, c1_v = scope.get_waveform_numpy(channel="C1", str_length=8000)  ## interchanged melbadry
    #     c2_t, c2_v = scope.get_waveform_numpy(channel="C2", str_length=8000)  ## interchanged melbadry
    #
    #     # Save the spectra of both inband and out of band
    #     if save_data:
    #         # channel1 = np.asarray([c1_t, c1_v])
    #         # channel1 = channel1.transpose()
    #         # np.savetxt(datapath + dirpath + "/channel_1_" + str(amplitude) + ".csv", channel1, delimiter=";")
    #
    #         channel2 = np.asarray([c2_t, c2_v])
    #         channel2 = channel2.transpose()
    #         np.savetxt(datapath + dirpath + "/lin_channel_2_" + str(amplitude) + ".csv", channel2, delimiter=";")
    # #### end of rerun with predistortion##

    # Save the SNR and SFDR together with the plots
    for i in range(len(time_PM)):
        if time_PM[i] < 0:
            time_PM[i] += 360

    for i in range(1,len(sweep_range)-1):
        if abs(time_PM[i] - time_PM[i+1] ) >4:
            if abs(time_PM[i] - time_PM[i-1] ) >4:
                time_PM[i] = (time_PM[i-1]+time_PM[i+1])/2
    if abs(time_PM[0]-time_PM[1]) > 4:
        time_PM[0] = time_PM[1]
    if abs(time_PM[i+1] - time_PM[i]) > 4:
        time_PM[i+1] = time_PM[i]

    # for i in range(1,len(sweep_range)-1):
    #     if abs(time_PM_lin[i] - time_PM_lin[i+1] ) >2:
    #         if abs(time_PM_lin[i] - time_PM_lin[i-1] ) >2:
    #             time_PM_lin[i] = (time_PM_lin[i-1]+time_PM_lin[i+1])/2
    # if abs(time_PM_lin[0]-time_PM_lin[1]) > 2:
    #     time_PM_lin[0] = time_PM_lin[1]
    # if abs(time_PM_lin[i+1] - time_PM_lin[i]) > 2:
    #     time_PM_lin[i+1] = time_PM_lin[i]

    if save_data:
        np.savetxt(datapath + dirpath + "/outputCode.csv", sweep_range, delimiter=";")
        #np.savetxt(datapath + dirpath + "/outputCode_lin.csv", sweep_range_lin, delimiter=";")
        np.savetxt(datapath + dirpath + "/time_AM.csv", time_AM, delimiter=";")
        np.savetxt(datapath + dirpath + "/time_PM.csv", time_PM, delimiter=";")
        np.savetxt(datapath + dirpath + "/Current_VDDmw.csv", Curr_VDDmw, delimiter=";")
        # np.savetxt(datapath + dirpath + "/time_AM_lin.csv", time_AM_lin, delimiter=";")
        # np.savetxt(datapath + dirpath + "/time_PM_lin.csv", time_PM_lin, delimiter=";")
        # np.savetxt(datapath + dirpath + "/Current_VDDmw_lin.csv", Curr_VDDmw_lin, delimiter=";")

if save_data:
    # plt.rcParams['text.usetex'] = True
    fonting =16

    fig, ax1 = plt.subplots()
    time_current = time_AM/(50*0.004)*1e3 # convert to mA
    time_current = time_current / time_current[-1]
    ax1.plot(sweep_range, time_current, label='AM-AM', color='blue', linewidth=3)
    #ax1.plot(sweep_range, time_AM_lin, label='Linearized AM', color='blue', linewidth=3,linestyle=':')
    #ax1.plot(sweep_range,  ideal_AM, label='Ideal AM', color='yellow', linewidth=3, linestyle=':')
    #ax1.legend(loc='lower right', fontsize=12)
    ax1.set_xlabel("AM Code", fontweight='bold', fontname='Arial', fontsize=fonting)  # <-- added font styling



    ax1.set_ylabel("Normalized AM-AM", color="blue", fontweight='bold', fontname='Arial', fontsize=fonting)  # <-- added font styling

    ax1.tick_params(axis='x', colors='k', labelsize=fonting)  # <-- added tick colors and size






    ax2 = ax1.twinx()
    ax2.plot(sweep_range,  time_PM - time_PM[-1], '-o', label='AM-PM', color='red', linewidth=2, markersize=5)
    #ax2.plot(sweep_range, time_PM_lin, label='Linearized PM', color='red',linestyle=':')
    #ax2.legend(loc='lower right', fontsize=12)

    # ## plot a linear fit
    sweep_range = sweep_range.flatten()
    time_PM = time_PM.flatten()
    fit_coeffs = np.polyfit(sweep_range, time_PM - time_PM[-1], deg=1)
    best_fit_line = np.poly1d(fit_coeffs)

    # Evaluate line over a dense x-range for smooth plotting
    sweep_dense = np.linspace(sweep_range.min(), sweep_range.max(), 500)
    ax2.plot(sweep_dense, best_fit_line(sweep_dense), color='darkred', linestyle='--', linewidth=2)

    ax2.set_ylabel("PM [Degrees]", color= "red", fontsize=fonting)
    ax2.tick_params(axis ='y', labelcolor = "red", labelsize=fonting)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=fonting)

    # Style spines
    for ax in [ax1, ax2]:  # ← ADDED (loop for consistency)
        for spine in ax.spines.values():
            spine.set_linewidth(3)

    # Style all tick labels
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():  # ← CHANGED (consolidated styling)
        label.set_fontname('Arial')
        label.set_fontweight('bold')

    ax1.set_facecolor('white')
    ax1.grid(True)
    ax1.set_facecolor('white')  # <-- added background color like original

    ax1.grid(True)  # <-- added grid enabled on primary axis (like plt.grid(True))



    #plt.title('AM-AM AM-PM')
    plt.show()

    fig.savefig(datapath + dirpath + "/" + "AM_PM.png", dpi=300)











