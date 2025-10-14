import win32com.client
import numpy as np

class MAUI:
    def __init__(self, resource_name="LeCroy.ActiveDSOCtrl.1"):
        self.resource_name = resource_name
        self.scope = None

    def __enter__(self):
        print(f"Connecting to MAUI scope via {self.resource_name}")
        self.scope = win32com.client.Dispatch(self.resource_name)
        self.scope.MakeConnection("IP:169.254.250.104")  # or use actual IP or USB if configured
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.scope:
            self.scope.Disconnect()

    def write(self, cmd):
        self.scope.WriteString(cmd, 1)

    def read(self, num):
        val = self.scope.ReadString(num)
        return val

    def query(self, cmd, str_length = 80):
        self.scope.WriteString(cmd, 1)
        return self.scope.ReadString(str_length)

    def idn(self):
        return self.query("*IDN?")

    def reset(self):
        self.write("*RST")

    def trigger(self):
        self.write("ARM")

    def wait(self):
        self.write("WAIT")

    def stop(self):
        self.write("STOP")

    def set_timebase(self, value):
        self.write(f"TDIV {value}")

    def set_trigger_mode(self, mode):
        self.write(f"TRMD {mode}")

    def set_trigger_level(self, channel, level):
        self.write(f"{channel}:TRLV {level}")

    def set_vertical_scale(self, channel, volts_per_div):
        self.write(f"{channel}:VDIV {volts_per_div}")

    def get_waveform_numpy(self, channel="C1", str_length=80):
        self.write("CHDR OFF;CORD LO;CFMT DEF9,BYTE,BIN")
        self.write("WFSU SP,0,NP,0,FP,0,SN,0")

        self.write(f"{channel}:WF? ALL")
        raw = self.scope.ReadBinary(str_length)

        volts_per_div = float(self.query(f"{channel}:VDIV?"))
        offset_volts = float(self.query(f"{channel}:OFST?"))
        time_div = float(self.query("TDIV?"))
        dt = float(self.query(f"{channel}:INSP? HORIZ_INTERVAL, FLOAT").split(":")[1].strip().replace('"', ''))
        # sample_rate = 40e9
        print("CHDR",self.query("CHDR?")) #"OFF"
        print("CORD",self.query("CORD?")) #"LO"
        print("CFMT",self.query("CFMT?")) #"DEF9,BYTE,BIN"
        print("WFSU", self.query("WFSU?")) # "SP,0,NP,0,FP,0,SN,0"
        # print(self.query("C1:INSP? HORIZ_INTERVAL"))
        byte_data = np.frombuffer(raw[364:], dtype=np.int8)
        volt_data = byte_data * (volts_per_div / 25.0) + offset_volts
        # dt = 1 / sample_rate
        time_data = np.arange(len(volt_data)) * dt
        return time_data, volt_data

    def auto_setup(self):
        self.write("ASET")

    def set_to_default(self):
        self.scope.ExecuteCommand("app.SetToDefaultSetup")