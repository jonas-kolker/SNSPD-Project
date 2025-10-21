from tkinter import *
from tkinter import ttk

def run():
    print(N_val.get())
    print(num_samples_val.get())
    print(num_loops_val.get())

root = Tk()
root.title("Jitter Experiment")

mainFrame = ttk.Frame(root, padding=(3,3,12,12))
mainFrame.grid( column=0, row=0, sticky=(N,W,E,S))

# Specify number of waveforms to acquire per sequence 
N_val = StringVar()
N_entry = ttk.Entry(mainFrame, width=7, textvariable=N_val)
N_entry.grid(row=1, column=2, sticky=(W,E))
ttk.Label(mainFrame, text="Number of segments per sequence").grid(row=1, column=1, sticky=E)

# Specify number of samples per waveform acquisition
num_samples_val = StringVar()
num_samples_entry = ttk.Entry(mainFrame, width=7, textvariable=num_samples_val)
num_samples_entry.grid(row=2, column=2, sticky=(W,E))
ttk.Label(mainFrame, text="Number of samples per segment").grid(row=2, column=1, sticky=E)

# Specify number of sequence 
num_loops_val = StringVar()
num_loops_entry = ttk.Entry(mainFrame, width=7, textvariable=num_loops_val)
num_loops_entry.grid(row=3, column=2, sticky=(W,E))
ttk.Label(mainFrame, text="Number of sequences").grid(row=3, column=1, sticky=E)

# Run experiment
ttk.Button(mainFrame, text="Run", command=run).grid(row=4, column=2, sticky=S)

# Allow adjusting window size and include padding
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
mainFrame.columnconfigure(2, weight=1)
for child in mainFrame.winfo_children():
    child.grid_configure(padx=5, pady=5)
N_entry.focus()

root.mainloop()