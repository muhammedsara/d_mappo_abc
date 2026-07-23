"""
Logging utilities for D-MAPPO-ABC
"""

import os
import datetime


class Logger:
    """Simple file logger"""
    
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filepath = os.path.join(log_dir, f"log_{timestamp}.txt")

    def log(self, message):
        """Write message to log file"""
        with open(self.filepath, "a") as f:
            f.write(message + "\n")
