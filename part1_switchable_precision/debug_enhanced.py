#!/usr/bin/env python3
"""
Enhanced debug wrapper to identify why the process is being killed at iteration 242
"""

import signal
import sys
import subprocess
import time
import psutil
import os
import threading
import resource

def monitor_process(pid, stop_event):
    """Continuously monitor a process's resource usage"""
    try:
        process = psutil.Process(pid)
        max_memory = 0
        iteration = 0

        with open('resource_log.txt', 'w') as f:
            f.write("Time,Memory_GB,CPU_%,Threads\n")

            while not stop_event.is_set():
                try:
                    mem_gb = process.memory_info().rss / 1024**3
                    cpu_pct = process.cpu_percent()
                    threads = process.num_threads()

                    max_memory = max(max_memory, mem_gb)

                    # Log every 5 seconds
                    if iteration % 5 == 0:
                        f.write(f"{time.time()},{mem_gb:.2f},{cpu_pct},{threads}\n")
                        f.flush()

                        # Check for suspicious patterns
                        if iteration == 240:  # Near the kill point
                            print(f"\n!!! Approaching iteration 242 kill point !!!")
                            print(f"Current memory: {mem_gb:.2f} GB")
                            print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

                    iteration += 1
                    time.sleep(1)

                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    print(f"Monitor error: {e}")

        print(f"\nPeak memory usage: {max_memory:.2f} GB")

    except Exception as e:
        print(f"Monitor thread error: {e}")

def check_slurm_limits():
    """Check SLURM job limits"""
    print("\n=== SLURM Configuration ===")

    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        print(f"Job ID: {job_id}")

        # Try to get job info
        try:
            result = subprocess.run(
                ['scontrol', 'show', 'job', job_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout

                # Parse key limits
                for line in output.split('\n'):
                    if 'TimeLimit=' in line:
                        print(f"Time Limit: {line.split('TimeLimit=')[1].split()[0]}")
                    if 'RunTime=' in line:
                        print(f"Run Time: {line.split('RunTime=')[1].split()[0]}")
                    if 'mem=' in line.lower():
                        print(f"Memory: {line}")

        except Exception as e:
            print(f"Could not get SLURM info: {e}")
    else:
        print("Not running under SLURM")

def check_system_limits():
    """Check system resource limits"""
    print("\n=== System Resource Limits ===")

    limits = [
        ('CPU time', resource.RLIMIT_CPU),
        ('Max memory', resource.RLIMIT_AS),
        ('Data segment', resource.RLIMIT_DATA),
        ('Stack size', resource.RLIMIT_STACK),
        ('Core file size', resource.RLIMIT_CORE),
        ('Processes', resource.RLIMIT_NPROC),
    ]

    for name, limit in limits:
        try:
            soft, hard = resource.getrlimit(limit)
            soft_str = 'unlimited' if soft == resource.RLIM_INFINITY else f"{soft:,}"
            hard_str = 'unlimited' if hard == resource.RLIM_INFINITY else f"{hard:,}"
            print(f"{name:15} - Soft: {soft_str:15} Hard: {hard_str}")
        except:
            pass

def analyze_kill_pattern():
    """Analyze the kill pattern - always at iteration 242"""
    print("\n=== Kill Pattern Analysis ===")
    print("Process consistently killed at iteration 242")
    print("Time to kill: ~90 seconds")
    print("This suggests:")
    print("  1. CPU time limit (90 seconds is suspicious)")
    print("  2. Iteration-based watchdog")
    print("  3. Memory threshold at specific data point")

    # Calculate expected time
    iterations_per_second = 242 / 90  # ~2.7 it/s
    total_time_needed = 2000 / iterations_per_second / 60
    print(f"\nEstimated time for 2000 iterations: {total_time_needed:.1f} minutes")
    print(f"You need at least {total_time_needed:.1f} minutes of runtime")

def main():
    # Check limits first
    check_system_limits()
    check_slurm_limits()
    analyze_kill_pattern()

    print("\n=== Starting Process with Monitoring ===\n")

    # Start the subprocess
    proc = subprocess.Popen(
        [sys.executable, "main_qat.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Start monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_process,
        args=(proc.pid, stop_event)
    )
    monitor_thread.start()

    start_time = time.time()
    last_line = ""
    iteration_count = 0

    try:
        # Read output line by line
        for line in proc.stdout:
            print(line, end='')
            last_line = line.strip()

            # Track iterations
            if "QAT:" in line and "%" in line:
                try:
                    # Extract iteration number
                    parts = line.split('|')[0]
                    if '/' in parts:
                        current = int(parts.split()[1].split('/')[0])
                        iteration_count = current
                except:
                    pass

        # Wait for process to complete
        return_code = proc.wait()
        elapsed = time.time() - start_time

    finally:
        stop_event.set()
        monitor_thread.join(timeout=2)

    print("\n" + "="*60)
    print(f"Process ended with return code: {return_code}")
    print(f"Last iteration reached: {iteration_count}")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Last output: {last_line}")

    # Interpret return code
    if return_code == -9 or return_code == 137:
        print("\n!!! SIGKILL detected !!!")
        print("Since this happens consistently at ~90 seconds/iteration 242:")
        print("1. Check if there's a 90-second CPU time limit:")
        print("   ulimit -t")
        print("2. For SLURM, request more time:")
        print("   #SBATCH --time=01:00:00")
        print("3. Check cgroup limits:")
        print("   cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        print("   cat /sys/fs/cgroup/memory/memory.limit_in_bytes")

    print("\nCheck 'resource_log.txt' for detailed resource usage over time")
    print("="*60)

if __name__ == "__main__":
    main()