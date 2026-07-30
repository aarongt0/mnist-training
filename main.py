import numpy as np
import sys
import tkinter as tk
from tkinter import ttk
from tkterminal import Terminal
import model
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
from matplotlib.figure import Figure
import sv_ttk

MIN_LR = 0.01
MAX_LR = 2.0

root = tk.Tk()

root.title("MNIST Training")
root.geometry("1000x1000")

class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)

    def flush(self):
        pass

total_accuracies = []
total_costs = []
total_epochs = 0

lr = 0.5
plot_x = list(range(total_epochs))
plot_y = total_costs

fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot(plot_x, plot_y)
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Cost")

canvas = FigureCanvasTkAgg(fig, master=root)   
canvas.draw()                                   
canvas.get_tk_widget().pack(side="bottom", fill=tk.BOTH, expand=False)

line, = ax.plot([], [])

def refresh_plot():
    line.set_data(range(len(total_costs)), total_costs)
    ax.relim()            
    ax.autoscale_view()   
    canvas.draw_idle()

def try_run_epochs():
    global running
    if running:
        return
    n_epochs = int(epoch_entry.get())
    if n_epochs < 1:
        return
    running = True
    _train_one(0, n_epochs)

def _train_one(i, n_epochs):
    global total_epochs, running, lr
    if i >= n_epochs:
        running = False
        return
    acc, cost = model.run_epoch(lr, i)
    total_accuracies.append(acc)
    total_costs.append(cost)
    total_epochs += 1
    print(f"Epoch {total_epochs} accuracy: {acc * 100:.2f}%")
    refresh_plot()
    root.after(1, lambda: _train_one(i + 1, n_epochs))

def lr_changed(value):
    global lr
    if (running): return
    lr = float(value)
    lr_label.config(text=f"Learning Rate: {float(value):.4f}")

title_label = ttk.Label(root, text="Train a model on the MNIST Database!", font=("Arial", 24, "bold"))
title_label.pack(side="top")

slider = ttk.Scale(
    root,
    from_=MIN_LR,      
    to=MAX_LR,       
    orient="horizontal", 
    length=300,     
    command=lr_changed 
)
slider.pack(pady=10, side="top")

lr_label = ttk.Label(root, text="Learning Rate: 0.01")
lr_label.pack(side="top")

terminal = Terminal(root)
terminal.shell = False
terminal.pack(side="left")

sys.stdout = TextRedirector(terminal)

epoch_label = ttk.Label(root, text="Epochs:")
epoch_label.pack(side="top")

epoch_entry = ttk.Entry(root, width=30)
epoch_entry.pack(side="top")

running: bool = False

def save_params():
    global running
    if running:
        return
    model.save_params(model.weights, model.biases)

def delete_params():
    global running
    if running:
        return
    model.delete_params()

def reset_params():
    global running, total_epochs, total_accuracies, total_costs
    if running:
        return
    model.reset_params()
    total_epochs = 0
    total_costs = []
    total_accuracies = []

def test_model():
    global running
    if running:
        return
    running = True
    acc = model.run_test_set()
    print(f"Test set accuracy = {acc * 100:.2f}%")
    running = False

epoch_send = ttk.Button(root, text="Train", command=try_run_epochs)
epoch_send.pack()

model_test = ttk.Button(root, text="Test Model", command=test_model)
model_test.pack()

params_reset = ttk.Button(root, text="Reset Parameters", command=reset_params)
params_reset.pack(side="bottom", pady=10)

params_save = ttk.Button(root, text="Save/Overwrite Paramaters", command=save_params)
params_save.pack(side="bottom", pady=10)

params_del = ttk.Button(root, text="Delete Paramaters", command=delete_params)
params_del.pack(side="bottom", pady=10)

sv_ttk.set_theme("dark", root)

print("")

root.mainloop()