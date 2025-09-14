#!/usr/bin/env python3
"""
Quick fix for memory leak issue - disable gradient checkpointing
"""

import sys
import os

# Add shared folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Import and modify the model
from models import QATGPT2

# Monkey patch to disable gradient checkpointing
original_init = QATGPT2.__init__

def patched_init(self, config):
    original_init(self, config)
    self.use_gradient_checkpointing = False  # Disable gradient checkpointing
    print("!!! Gradient checkpointing DISABLED to prevent memory leak !!!")

QATGPT2.__init__ = patched_init

# Now run the main script
if __name__ == "__main__":
    exec(open("main_qat.py").read())