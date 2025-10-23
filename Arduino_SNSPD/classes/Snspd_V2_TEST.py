# -*- coding: utf-8 -*-
"""
Created on 14 June 2024

@author: Giovanni Carboni

Inpired by Luc Enthoven
"""

import serial
import time
import logging





class Snspd(object):
    """
    Todo: Fix ordering of writing the registers (MGstruct is last, not first)
    """
    
    # Dictonary to order the registers
    registers = ["DCcompensate",
                 "DFBamp",
                 "DSNSPD",
                 "DAQSW",
                 "VRL",
                 "Dbias_NMOS",
                 "DBias_internal",
                 "Dbias_fb_amp",
                 "Dbias_comp",
                 "Dbias_PMOS",
                 "Dbias_ampNMOS",
                 "Ddelay",
                 "Dcomp",
                 "Analoga",
                 "Dbias_ampPMOS",
                 "DCL",
                 "Dbias_ampn1",
                 "Dbias_ampn2"]
    
    Reg_dict = {}
    reverse_order=False

    for i, reg in enumerate(registers):
        if reverse_order==True:
            Reg_dict[reg] = [33-i-1] #Dbias_ampn2 has index 0
        else:
            Reg_dict[reg] = [i] #DCcompensate has index 0
    #print(Reg_dict)

    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM15'
    ser.timeout = 1
    opcode = 0x00
    reg = [0]*33
    
    def __init__(self, com):
        self.ser.port = com
        
        
    def __enter__(self):
        logging.debug("Opening Snspd on COM port ", self.ser.port)
        self.ser.open()
        time.sleep(2)
        print("serial port open")
        return self        
    
        
    def __exit__(self, type, value, traceback):
        self.ser.close()
        print('serial closed')
        
    ### --------------------------- Functions ------------------------ ###

    """ Arduino operation codes
    opcode == 0x04
                    Write registers without debug and read back
                    returns one byte
                    0xFF for success in comparison
                    0x00 for check failed
    opcode == 0x05
                    Write registers with debug and read back
                    it transmits couples of bytes
                    first one is the sent
                    second one is the received
                    then it adds the check bytes as above
    """

    def TX_reg_debug(self):
        self.opcode=0xFF #debug byte
        data = bytearray([self.opcode] + self.reg)
        return self.send_register_debug(data)
    
    def TX_reg(self):
        self.opcode=0xFF #termination byte
        data = bytearray([0] + self.reg + [self.opcode])
        return self.send_register(data)
    
   

    def send_register(self, write_word):
    
        if len(write_word)==35:
            print("1st register sent, 2nd register received\n")
            print("Sent:                        ", write_word[1:])
            #write_word = bytearray([0x01, 0xFF])
            
            self.ser.write(write_word)
            time.sleep(2)
            #self.ser.read(1)
            #echo = self.ser.read(33)
            #print("echo:                                  ", echo)
        
            time.sleep(5)
            from_ard = self.ser.read(40)
            print("Received:                              ", from_ard)
            
            if write_word[1:] == from_ard:
                print("Succesful connection")
                return True
            else:
                print("Sent/received don't match")
                return False            
            
            
        else:
            logging.error("34 bytes should be passed to the write_register function")
            return False

####################################################################
    def set_register(self, 
                     DCcompensate=0,
                     DFBamp=0,
                     DSNSPD=0,
                     DAQSW=0,
                     VRL=0,
                     Dbias_NMOS=0,
                     DBias_internal=True,
                     Dbias_fb_amp=0,
                     Dbias_comp=0,
                     Dbias_PMOS=0,
                     Dbias_ampNMOS=0,
                     Ddelay=0,
                     Dcomp=0,
                     Analoga='',
                     Dbias_ampPMOS=0,
                     DCL=0,
                     Dbias_ampn1=0,
                     Dbias_ampn2=0):
        
        self.set_DCcompensate(DCcompensate=DCcompensate)
        self.set_DFBamp(DFBamp=DFBamp)
        self.set_DSNSPD(DSNSPD=DSNSPD)
        self.set_DAQSW(DAQSW=DAQSW)
        self.set_VRL(VRL=VRL)
        self.set_Dbias_NMOS(Dbias_NMOS=Dbias_NMOS)
        self.set_DBias_internal(DBias_internal=DBias_internal)
        self.set_Dbias_fb_amp(Dbias_fb_amp=Dbias_fb_amp)
        self.set_Dbias_comp(Dbias_comp=Dbias_comp)
        self.set_Dbias_PMOS(Dbias_PMOS=Dbias_PMOS)
        self.set_Dbias_ampNMOS(Dbias_ampNMOS=Dbias_ampNMOS)
        self.set_Ddelay(Ddelay=Ddelay)
        self.set_Dcomp(Dcomp=Dcomp)
        self.set_Analoga(Analoga=Analoga)
        self.set_Dbias_ampPMOS(Dbias_ampPMOS=Dbias_ampPMOS)
        self.set_DCL(DCL=DCL)
        self.set_Dbias_ampn1(Dbias_ampn1=Dbias_ampn1)
        self.set_Dbias_ampn2(Dbias_ampn2=Dbias_ampn2)

###-----------------### Registers functions ###----------------------###

#      ---- Set DCcompensate ----
    def set_DCcompensate(self, DCcompensate=0):
        val=DCcompensate
        pos=self.pos("DCcompensate")
        if val>7 or val<0:
            print("val for DCcompensate is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF
        return self.reg

#      ---- Set DFBamp ----
    def set_DFBamp(self, DFBamp=0):
        val=DFBamp
        pos=self.pos("DFBamp")
        if val>15 or val<0:
            print("val for DFBamp is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set DSNSPD ----
    def set_DSNSPD(self, DSNSPD=0):
        val=DSNSPD
        pos=self.pos("DSNSPD")
        if val>127 or val<0:
            print("val for DSNSPD is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = (val>>0)&0xFF

        return self.reg

#      ---- Set DAQSW ----
    
    def set_DAQSW(self, DAQSW=0):
        val=DAQSW
        pos=self.pos("DAQSW")
        if val>127 or val<0:
            print("val for DAQSW is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg
    
#      ---- Set VRL ----
    def set_VRL(self, VRL=0):
        val=VRL
        pos=self.pos("VRL")
        if val>31 or val<0:
            print("val for Load Resistance (VRL) is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg
    
#      ---- Set Dbias_NMOS ----
    def set_Dbias_NMOS(self, Dbias_NMOS=0):
        val=Dbias_NMOS
        pos=self.pos("Dbias_NMOS")
        if val>255 or val<0:
            print("val for Dbias_NMOS is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set DBias_internal ----
    def set_DBias_internal(self, DBias_internal):
        val=DBias_internal
        pos=self.pos("DBias_internal")
        if val>1 or val<0:
            print("Error with 'internal current' bit, please check")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF
            #self.reg[pos] = 255&0xFF

        return self.reg

#      ---- Set Dbias_fb_amp ----
    def set_Dbias_fb_amp(self, Dbias_fb_amp=0):
        val=Dbias_fb_amp
        pos=self.pos("Dbias_fb_amp")
        if val>127 or val<0:
            print("val for Dbias_fb_amp is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dbias_comp ----
    def set_Dbias_comp(self, Dbias_comp=0):
        val=Dbias_comp
        pos=self.pos("Dbias_comp")
        if val>127 or val<0:
            print("val for Dbias_comp is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dbias_PMOS ----
    def set_Dbias_PMOS(self, Dbias_PMOS=0):
        val=Dbias_PMOS
        pos=self.pos("Dbias_PMOS")
        if val>200 or val<0:
            print("val for Dbias_PMOS is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dbias_ampNMOS ----
    def set_Dbias_ampNMOS(self, Dbias_ampNMOS=0):
        val=Dbias_ampNMOS
        pos=self.pos("Dbias_ampNMOS")
        if val>127 or val<0:
            print("val for Dbias_ampNMOS is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Ddelay ----
    def set_Ddelay(self, Ddelay=0):
        val=Ddelay
        pos=self.pos("Ddelay")
        if val>127 or val<0:
            print("val for Ddelay is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dcomp ----
    def set_Dcomp(self, Dcomp=0):
        val=Dcomp
        pos=self.pos("Dcomp")
        if val>15 or val<0:
            print("val for Dcomp is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Analoga ----
    def set_Analoga(self, Analoga):
        pos=self.pos("Analoga")
        match Analoga:
            case 'None':
                self.reg[pos] = 0&0x00
            case 'Vref':
                self.reg[pos] = 1&0xFF
            case 'Vamp':
                self.reg[pos] = (1<<1)&0xFF
            case 'Vcomp':
                self.reg[pos] = (1<<2)&0xFF
            case _:
                print("value for Source follower output is incorrect, choose between None, Vref, Vamp, Vcomp")
                self.reg[pos] = 0x00
        #print(format(self.reg[pos], '08b'))
        return self.reg

#      ---- Set Dbias_ampPMOS ----
    def set_Dbias_ampPMOS(self, Dbias_ampPMOS=0):
        val=Dbias_ampPMOS
        pos=self.pos("Dbias_ampPMOS")
        if val>127 or val<0:
            print("val for Dbias_ampPMOS is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set DCL ----
    def set_DCL(self, DCL=0):
        val=DCL
        pos=self.pos("DCL")
        if val>15 or val<0:
            print("val for DCL is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dbias_ampn1 ----
    def set_Dbias_ampn1(self, Dbias_ampn1=0):
        val=Dbias_ampn1
        pos=self.pos("Dbias_ampn1")
        if val>127 or val<0:
            print("val for Dbias_ampn1 is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

#      ---- Set Dbias_ampn2 ----
    def set_Dbias_ampn2(self, Dbias_ampn2=0):
        val=Dbias_ampn2
        pos=self.pos("Dbias_ampn2")
        if val>127 or val<0:
            print("val for Dbias_ampn2 is too small or to large")
            self.reg[pos] = 0x00
        else:
            self.reg[pos] = val&0xFF

        return self.reg

    def pos(self, string): #obtain position index for the register
        return self.Reg_dict.get(string,[])[0]

 
    def print_reg(self):
        regsize = len(self.reg)
        for i in range(regsize):
            print(i, self.reg[i])
        

    """
    pbin = []
                for i in range(34):
                    pbin.append(format(write_word[i], '08b'))
                print(pbin)
    """
    
if __name__ == "__main__":    
    # with Snspd('COM12') as snspd:
    #     x = snspd.set_register()
    #     print(x)
    #     ret_bool = snspd.tx_reg_debug()
    #     print(ret_bool)
    #     ret_bool = snspd.tx_reg_checked()
    #     print(ret_bool)
    pass