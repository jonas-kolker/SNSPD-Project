# -*- coding: utf-8 -*-
import serial

from classes.Snspd_V2_TEST import Snspd
from classes.SMU_2636B import SMU_2636B

import numpy as np
import time
import matplotlib.pyplot as plt


import pyvisa as visa

import sys
from sys import exit


plt.style.use("ggplot")
rm = visa.ResourceManager()

with open("COM_ports.txt") as fin: #the file contains the COM port of arduino and the VISA address of the SMU
    arduinoPort, SMU_addr = fin.read().splitlines()

#%%

with Snspd(arduinoPort) as snspd:

    SNSPD_Dcode =20
    RAQSW = 40
    Load = 8

    D_code = int(round(SNSPD_Dcode*5/7))
    snspd.set_register( DCcompensate=2, #miller capacitance
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
                        DCL=8, #FB amplifier load capacitance
                        Dbias_ampn1=D_code*2,
                        Dbias_ampn2=D_code)
    print("Dbias_ampn2 = ", D_code)

    
    check = snspd.TX_reg()

    """
    for Dbias in range(1, 30):
        print("Dbias = ", Dbias)
        mod_reg(Dbias)
    """