# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:42:19 2021

@author: lucenthoven
"""

import time
import logging
import pyvisa
from classes.DT670_diode import DT670_diode
import numpy as np

logger = logging.getLogger('SMU_class')
logger.setLevel(logging.DEBUG)


class SMU_2636B(object):
    """
    SMU which is to be connected:
        Temperature Diode on ch A
        Source follower on ch B
    The setup for this is handled in the init_SMU() function
    """
    
    name = ''
    rm = pyvisa.ResourceManager()
        
    
    def __init__(self, name):
        self.name = name
           
    def __enter__(self):
        devices = self.rm.list_resources()
        if self.name in devices:
            logging.debug("Connecting to SMU with address ", self.name)
            self.instr = self.rm.open_resource(self.name)
            self.instr.timeout = 25000
            self.init_SMU()
            return self
        else:
            logging.error("Address for SMU is not found")
            
    def __exit__(self, type, value, traceback):
        self.instr.close()

    def open(self):
        self.__enter__()
        
    def close(self):
        self.__exit__()
        
    def list_resources(self):
        self.rm.list_resources()

    def reset(self):
        self.instr.write("*RST")
        self.instr.write("smua.reset()")
        self.instr.write("smub.reset()")
         
    def init_SMU_chA_I(self):
        self.instr.write("beeper.enable = 1")
        self.instr.write("smua.reset()")
        self.instr.write("smub.reset()")
        
        self.instr.write("format.asciiprecision = 10")
        
        '''
        Set SMU A to 4 wire measurement for the temperature diode
        Set the level and range to 10uA
        Set max output voltage to 2 V
        '''
        
        # Set SMU A channel to 4 wire measurement
        self.instr.write("smua.sense = smua.SENSE_REMOTE")
        self.instr.write("smua.source.func = smua.OUTPUT_DCAMPS")
        self.instr.write("smua.source.rangei = 10e-6")
        self.instr.write("smua.source.leveli = 10e-6")
        
        self.instr.write("smua.source.limitv = 2")
        self.instr.write("smua.source.rangev = 2")

    def init_clear(self):
        self.instr.write("*CLS")
    def errorqueue_clear(self):
        self.instr.write("errorqueue.clear()")

    def init_SMU(self):
        self.instr.write("beeper.enable = 1")
        self.instr.write("smua.reset()")
        self.instr.write("format.asciiprecision = 10")
        
        '''
        Set SMU A to 4 wire measurement for the temperature diode
        Set the level and range to 10uA
        Set max output voltage to 2 V
        '''
        
        # Set SMU A channel to 4 wire measurement
        self.instr.write("smua.sense = smua.SENSE_REMOTE")
        self.instr.write("smua.source.func = smua.OUTPUT_DCAMPS")
        self.instr.write("smua.source.rangei = 10e-6")
        self.instr.write("smua.source.leveli = 10e-6")
        
        self.instr.write("smua.source.limitv = 2")
        self.instr.write("smua.source.rangev = 2")

        '''
        Set SMU B to 2 wire measurement for the SF reading
        Set the level to 1.3 mA and range to 10 mA
        Set max output voltage to 2 V
        '''
        # Set SMU B channel to 2(1) wire measurement
        self.instr.write("smub.reset()")
        self.instr.write("smub.sense = smub.SENSE_LOCAL")
        self.instr.write("smub.source.func = smub.OUTPUT_DCAMPS")
        self.instr.write("smub.source.rangei = 10e-3") #    10 mA
        self.instr.write("smub.source.leveli = 13e-4") #    1.3 mA
        self.instr.write("smub.source.limitv = 2") #        2 V
        self.instr.write("smub.source.rangev = 2") #        2 V
        self.instr.write("smub.measure.nplc = 1")

    def print_smu_info(self):
        ret = self.instr.query("*IDN?")
        print(ret)
        return ret
    
    def clear_buffer(self):
        self.instr.write("smua.nvbuffer1.clear()")
        self.instr.write("smua.nvbuffer2.clear()")
        self.instr.write("smub.nvbuffer1.clear()")
        self.instr.write("smub.nvbuffer2.clear()")

    def clear_display(self):
        self.instr.write("display.clear()")

    def enable_chA(self):
        self.instr.write("smua.source.output = smua.OUTPUT_ON")

    def disable_chA(self):
        self.instr.write("smua.source.output = smua.OUTPUT_OFF")

    def enable_chB(self):
        self.instr.write("smub.source.output = smub.OUTPUT_ON")

    def disable_chB(self):
        self.instr.write("smub.source.output = smub.OUTPUT_OFF")
     
    def meas_V_chA(self):
        ret = self.instr.query_ascii_values("print(smua.measure.v())")
        # print(ret)
        if type(ret) == list:
            return ret[-1]
        else:
            return ret
    
    def meas_V_chA2(self):
        self.instr.write("mybuffer = smuX.makebuffer(10)")
        self.instr.write("smua.measure.v(mybuffer)")
        time.sleep(0.3)
        ret = self.instr.query_ascii_values("printbuffer(0, 9, mybuffer)")
        if type(ret) == list:
            return ret[-1]
        else:
            return ret
    
    def set_I_chA(self, val=1e-6):
        self.instr.write("smua.source.leveli = {}".format(val)) 

    def init_2wireI_chA(self):
        self.instr.write("smua.sense = smua.SENSE_LOCAL")
        self.instr.write("smua.source.func = smua.OUTPUT_DCAMPS")
        self.instr.write("smua.source.rangei = 1e-5")
        self.instr.write("smua.source.leveli = 1e-6")
        self.instr.write("smua.source.limitv = 1")
        self.instr.write("smua.source.rangev = 1")
        self.instr.write("smua.measure.nplc = 10")
        
    def init_2wireV_chA(self): # modified by gcarboni
        self.instr.write("display.screen = display.SMUA")
        self.instr.write("display.smua.measure.func = display.MEASURE_DCVOLTS")
        self.instr.write("smua.sense = smua.SENSE_LOCAL")
        self.instr.write("smua.source.func = smua.OUTPUT_DCVOLTS")
        self.instr.write("smua.source.rangei = 1e-6")
        self.instr.write("smua.source.leveli = 1e-6")
        self.instr.write("smua.source.limiti = 1e-6")
        self.instr.write("smua.source.levelv = 1e-6")
        self.instr.write("smua.source.rangev = 1e-6")
        self.instr.write("smua.measure.nplc = 1")
    
    def init_2wireV_chB(self): # modified by gcarboni
        self.instr.write("display.screen = display.SMUB")
        self.instr.write("display.smub.measure.func = display.MEASURE_DCVOLTS")
        self.instr.write("smub.sense = smub.SENSE_LOCAL")
        self.instr.write("smub.source.func = smub.OUTPUT_DCVOLTS")
        #self.instr.write("smub.measure.autorangei = smub.AUTORANGE_ON")
        self.instr.write("smub.source.limiti = 1e-6")
        self.instr.write("smub.source.rangev = 1e-6")
        self.instr.write("smub.source.levelv = 1e-6")
        self.instr.write("smub.measure.nplc = 1")
    
    def init_2wireV_measI_chA(self): # modified by gcarboni
        self.instr.write("display.screen = display.SMUA")
        self.instr.write("display.smua.measure.func = display.MEASURE_DCAMPS")
        self.instr.write("smua.sense = smua.SENSE_LOCAL")
        self.instr.write("smua.source.func = smua.OUTPUT_DCVOLTS")
        self.instr.write("smua.source.limiti = 1e-6")
        self.instr.write("smua.source.rangev = 1e-6")
        self.instr.write("smua.source.levelv = 1e-6")
        self.instr.write("smua.measure.nplc = 1")
    
    def init_2wireV_measI_chB(self): # modified by gcarboni
        self.instr.write("display.screen = display.SMUB")
        self.instr.write("display.smub.measure.func = display.MEASURE_DCAMPS")
        self.instr.write("smub.sense = smub.SENSE_LOCAL")
        self.instr.write("smub.source.func = smub.OUTPUT_DCVOLTS")
        self.instr.write("smub.source.limiti = 1e-6")
        self.instr.write("smub.source.rangev = 1e-6")
        self.instr.write("smub.source.levelv = 1e-6")
        self.instr.write("smub.measure.nplc = 1")

    def init_2wireI_chB(self): # modified by gcarboni
        self.instr.write("display.screen = display.SMUB")
        self.instr.write("display.smub.measure.func = display.MEASURE_DCVOLTS")
        self.instr.write("smub.sense = smub.SENSE_LOCAL")
        self.instr.write("smub.source.func = smub.OUTPUT_DCAMPS")

        #self.instr.write("smub.source.limiti = 1e-6")
        #self.instr.write("smub.source.rangev = 10")
        #self.instr.write("smub.source.levelv = 1e-6")
        self.instr.write("smub.measure.nplc = 1")
    
    def display_chA(self):
        self.instr.write("display.screen = display.SMUA")
    
    def display_chB(self):
        self.instr.write("display.screen = display.SMUB")
        
    def display_both(self):
        self.instr.write("display.screen = display.SMUA_SMUB")

    def init_4wireI_chB_floating(self):
        self.instr.write("smub.sense = smub.SENSE_REMOTE")
        self.instr.write("smub.source.func = smub.OUTPUT_DCAMPS")
        self.instr.write("smub.source.rangei = 50e-3")
        # self.instr.write("smub.source.leveli = 1e-6")
        self.instr.write("smub.source.limitv = 500e-3")
        self.instr.write("smub.source.rangev = 1")
        self.instr.write("smub.measure.nplc = 25")
        
    def set_ilim_chB(self, limit=1e-3):
        self.instr.write("smub.source.limiti = {}".format(limit))
        
    def set_ilim_chA(self, limit=1e-3):
        self.instr.write("smua.source.limiti = {}".format(limit))
    
    def set_vlim_chA(self, limit = 2):
        self.instr.write("smua.source.limitv = {}".format(limit))

    def set_vlim_chB(self, limit = 2):
        self.instr.write("smub.source.limitv = {}".format(limit))
    
    def set_irange_chA(self, limit=20e-6):
        self.instr.write("smua.source.rangei = {}".format(limit))
    
    def set_irange_chB(self, limit=50e-3):
        self.instr.write("smub.source.rangei = {}".format(limit))

    def meas_IV_chB(self):
        ret_I = self.instr.query_ascii_values("print(smub.measure.i())")
        ret_V = self.instr.query_ascii_values("print(smub.measure.v())")
        print(ret_I)
        if type(ret_I) == list:
            return ret_I[-1], ret_V[-1]
        else:
            return ret_I, ret_V
    
    def meas_IV_chA(self):
        ret_I = self.instr.query_ascii_values("print(smua.measure.i())")
        ret_V = self.instr.query_ascii_values("print(smua.measure.v())")
        print(ret_I)
        if type(ret_I) == list:
            return ret_I[-1], ret_V[-1]
        else:
            return ret_I, ret_V
        
    def get_var(self, var="measI"):
        ret = self.instr.query_ascii_values("print("+var+")")
        return ret

    def meas_V_chA(self):
        ret = self.instr.query_ascii_values("print(smua.measure.v())")
        #print(ret)
        if type(ret) == list:
            return ret[-1]
        else:
            return ret
        
    def meas_V_chB(self):
        ret = self.instr.query_ascii_values("print(smub.measure.v())")
        #print(ret)
        if type(ret) == list:
            return ret[-1]
        else:
            return ret
        
    def meas_I_chA(self):
        ret = self.instr.query_ascii_values("print(smua.measure.i())")
        
        if type(ret) == list:
            return ret[-1]
        else:
            return ret
    
    def meas_I_chB(self):
        ret = self.instr.query_ascii_values("print(smub.measure.i())")
        # print(ret)
        if type(ret) == list:
            return ret[-1]
        else:
            return ret

    def set_I_chB(self, val=1e-6):
        string = "smub.source.leveli = {}".format(val)
        logging.debug(string)
        self.instr.write(string)

    def set_V_chB(self, val=1e-3):
        self.instr.write("smub.source.levelv = {}".format(val))

    def set_V_chA(self, val=1e-3):
        self.instr.write("smua.source.levelv = {}".format(val))

    def set_rangeV_chB(self, val=1):
        self.instr.write("smub.source.rangev = {}".format(val))

    def set_rangeI_chB(self, val=1e-3):
        self.instr.write("smub.source.rangei = {}".format(val))
 
    def set_rangeV_chA(self, val=1):
        self.instr.write("smua.source.rangev = {}".format(val))

    def set_rangeI_chA(self, val=1e-3):
        self.instr.write("smua.source.rangei = {}".format(val))
    
    
        
    def meas_temp(self, verbose=True):
        V_diode = self.meas_V_chA()
        if verbose:
            print("Temperature is: " + str(DT670_diode.linear_decode(V_diode)-273) + " °C" )
        return np.c_[DT670_diode.linear_decode(V_diode), V_diode]