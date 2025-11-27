import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import signal

# Path to your main.py script
SCRIPT_PATH = "main.py"

class HumanDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection Control")
        self.root.geometry("300x150")
        self.process = None

        # Start button
        self.start_btn = tk.Button(root, text="Start Detection", command=self.start_detection, width=20)
        self.start_btn.pack(pady=20)

        # Stop button
        self.stop_btn = tk.Button(root, text="Stop Detection", command=self.stop_detection, width=20)
        self.stop_btn.pack(pady=10)

        # Exit button
        self.exit_btn = tk.Button(root, text="Exit", command=self.exit_app, width=20)
        self.exit_btn.pack(pady=10)

    def start_detection(self):
        if self.process is None:
            # Run main.py in a new process
            self.process = subprocess.Popen(["python", SCRIPT_PATH])
            messagebox.showinfo("Info", "Detection started!")
        else:
            messagebox.showwarning("Warning", "Detection is already running!")

    def stop_detection(self):
        if self.process is not None:
            # Terminate the process
            os.kill(self.process.pid, signal.SIGTERM)
            self.process = None
            messagebox.showinfo("Info", "Detection stopped!")
        else:
            messagebox.showwarning("Warning", "Detection is not running!")

    def exit_app(self):
        if self.process is not None:
            os.kill(self.process.pid, signal.SIGTERM)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HumanDetectionGUI(root)
    root.mainloop()
