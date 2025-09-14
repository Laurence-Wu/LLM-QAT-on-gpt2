#!/usr/bin/env python3
"""
Debug wrapper for main_qat.py to catch and report why the process is being killed
"""

import signal
import sys
import subprocess
import time
import psutil
import os

def handler(signum, frame):
    print(f'\n{"="*60}')
    print(f'!!! Process received signal {signum} ({signal.Signals(signum).name}) !!!')
    print(f'{"="*60}')

    # Print current resource usage
    process = psutil.Process()
    print(f"Current memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"CPU percent: {process.cpu_percent()}%")
    print(f"Number of threads: {process.num_threads()}")

    sys.exit(1)

def monitor_resources():
    """Monitor and log resource usage"""
    process = psutil.Process()

    print("\n=== System Resource Information ===")
    print(f"Total system memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"CPU count: {psutil.cpu_count()}")

    if 'SLURM_JOB_ID' in os.environ:
        print(f"\nSLURM Job ID: {os.environ.get('SLURM_JOB_ID')}")
        print(f"SLURM Time Limit: {os.environ.get('SLURM_TIME_LIMIT', 'Not set')}")

    print("\n=== Starting Process Monitoring ===\n")

def main():
    # Register signal handlers for common termination signals
    for sig in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2]:
        try:
            signal.signal(sig, handler)
            print(f"Registered handler for {signal.Signals(sig).name}")
        except Exception as e:
            print(f"Could not register handler for signal {sig}: {e}")

    # Monitor initial resources
    monitor_resources()

    print("\nStarting main_qat.py with signal monitoring...\n")
    print("="*60)

    start_time = time.time()

    try:
        # Run the main script
        result = subprocess.run(
            [sys.executable, "main_qat.py"],
            check=False,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        print("\n" + "="*60)
        print(f"Process completed with return code: {result.returncode}")
        print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        if result.returncode == -9:
            print("!!! Process was killed with SIGKILL (kill -9) !!!")
            print("This usually indicates:")
            print("  1. OOM killer (out of memory)")
            print("  2. Manual kill by admin")
            print("  3. Job scheduler timeout")
        elif result.returncode == -15:
            print("!!! Process was terminated with SIGTERM !!!")
            print("This usually indicates:")
            print("  1. Job scheduler time limit reached")
            print("  2. System shutdown")
            print("  3. Manual termination")
        elif result.returncode == 137:
            print("!!! Exit code 137 - SIGKILL (9) received !!!")
        elif result.returncode == 143:
            print("!!! Exit code 143 - SIGTERM (15) received !!!")

    except KeyboardInterrupt:
        print("\n!!! Process interrupted by user (Ctrl+C) !!!")
    except Exception as e:
        print(f"\n!!! Unexpected error: {e} !!!")

    # Final resource check
    process = psutil.Process()
    print(f"\nFinal memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
    print("="*60)

if __name__ == "__main__":
    main()