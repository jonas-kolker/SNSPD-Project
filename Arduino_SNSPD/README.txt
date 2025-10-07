Front end is implemented through SLOW_SPI_COM_TEST.py
The file imports the "COM_ports.txt" file which contains information regarding:
1. the COMxx port for Arduino communication
2. Any other MAC address useful for remote control (SMUs, scope, etc...)
The reason for this is that if you use the same addresses in multiple files, and something changes in the USB configuration, you need to modify only once.


----------
"with Snspd(arduinoPort) as snspd:" opens the communication with the backend file ("from classes.Snspd_V2_TEST import Snspd")

The following portion of code is used to choose the desired settings in the cryo-CMOS

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

The file "shift register scheme.xlsx" can be helpful to understand the meaning of all the different variables

---------------- what's inside snspd.set_register? The fuction is part of the "classes.Snspd_V2_TEST.py" file
The function triggers a series of small other functions (one per register in the chip) that build the "self.reg" array, concatenating the binary values byte by byte.
more specifically
self.reg[pos] = val&0xFF creates a 8-bit word which represent the binary version of the digital value "val" and puts it in self.reg[pos].
"pos" is the index that matches the position of that byte in the shift registers on chip, and it is determined from a dictionary (it's actually a sort of look-up table) that is created at the beginning of this file.
--------------- 

at the end of this process, the frontend file call the following function:
snspd.TX_reg()
which simply builds a "bytearray" out of the array just built (it also adds a 0 at the beginning to compensate for 1bit mismatch in the Arduino code, and the command code at the end "[self.opcode]", I explain later what this means).

The function "send_register(data)" writes the bytearray on the serial "self.ser.write", waits a few seconds (these few seconds are the time needed by Arduino to decode, send to and receive from the CMOS) and then reads back from the serial "self.ser.read".

->> The criterion is that what is sent from py to Arduino must exactly match what is sent from Arduino to py after programming.


--------------- Arduino

Following code checks the UART for data. When data are sent form python (self.ser.write), this portion of code is triggered.

if (Serial.available()) {
        uint8_t receivedData[MAX_BUFFER_SIZE];
        int length = 0;	

	while (Serial.available()) {
            uint8_t byte = Serial.read();
            if (byte == 0xFF) break;   --> this is the [self.opcode] which works as termination character. It is unique since no other byte is "11111111"
            receivedData[length++] = byte;
            delayMicroseconds(10000);
        }


then the function "transmitAndReadSPI(receivedData, length)" sends the bytes serially to the chip and saves what the chip replies in a variable "response".
The content of this variable is sent back on the UART to python. This is the moment in which python backend is supposed to perform the "self.ser.read".

Please note that python works asynchronously wrt Arduino, so the time delay are calibrated to make everything work, there are no time references that make the system solid to variation of length or speed of the codes (this can be an improvement point).


-------------- Python backend
"self.ser.read" reads what Arduino is passing to it from the chip. the content is printed right away so that can be compared with what python send.
ANOTHER IMPROVEMENT POINT: now the comparison is done visually because the sent structure is a "bytearray" while the read structure is just a string (I guess). It would be nice if you could find a way to have a print that says "yes/no" (or something easy to read) that tells whether the programming was successful or not.























