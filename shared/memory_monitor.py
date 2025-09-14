"""
Memory monitoring utilities to detect and prevent memory leaks during training.
"""
import torch
import psutil
import gc
from functools import wraps
from contextlib import contextmanager
import warnings

class MemoryMonitor:
    """Monitor memory usage and detect potential memory leaks."""

    def __init__(self, threshold_mb=1000, verbose=True):
        """
        Initialize memory monitor.

        Args:
            threshold_mb: Memory increase threshold in MB to trigger warning
            verbose: Whether to print memory usage information
        """
        self.threshold_mb = threshold_mb
        self.verbose = verbose
        self.baseline_memory = self.get_memory_usage()
        self.peak_memory = self.baseline_memory

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024**2

    def get_memory_stats(self):
        """Get detailed memory statistics."""
        stats = {}
        if torch.cuda.is_available():
            stats['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            stats['free_mb'] = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2
            stats['fragmentation'] = stats['allocated_mb'] / stats['reserved_mb'] if stats['reserved_mb'] > 0 else 0
        else:
            process = psutil.Process()
            stats['rss_mb'] = process.memory_info().rss / 1024**2
            stats['vms_mb'] = process.memory_info().vms / 1024**2
        return stats

    def check_memory_leak(self, tag=""):
        """Check for potential memory leaks."""
        current_memory = self.get_memory_usage()
        increase = current_memory - self.baseline_memory

        # Update peak memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        if increase > self.threshold_mb:
            if self.verbose:
                warnings.warn(f"⚠️ Potential memory leak at {tag}: {increase:.2f} MB increase from baseline")
                stats = self.get_memory_stats()
                print(f"Memory stats: {stats}")

            # Attempt cleanup
            self.force_cleanup()

            # Check if cleanup helped
            new_memory = self.get_memory_usage()
            if self.verbose and new_memory < current_memory:
                print(f"✓ Cleanup recovered {current_memory - new_memory:.2f} MB")

        return increase

    def force_cleanup(self):
        """Force aggressive memory cleanup."""
        # Python garbage collection
        gc.collect()

        # PyTorch CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Clear gradient buffers if they exist
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.grad is not None:
                        obj.grad = None
                except:
                    pass

    def reset_baseline(self):
        """Reset the baseline memory measurement."""
        self.baseline_memory = self.get_memory_usage()

    def print_summary(self):
        """Print memory usage summary."""
        if self.verbose:
            current = self.get_memory_usage()
            print(f"\n📊 Memory Summary:")
            print(f"  Baseline: {self.baseline_memory:.2f} MB")
            print(f"  Current: {current:.2f} MB")
            print(f"  Peak: {self.peak_memory:.2f} MB")
            print(f"  Total increase: {current - self.baseline_memory:.2f} MB")


def monitor_memory(threshold_mb=500):
    """
    Decorator to monitor memory usage of a function.

    Args:
        threshold_mb: Memory increase threshold in MB to trigger warning
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor(threshold_mb=threshold_mb, verbose=True)

            try:
                result = func(*args, **kwargs)
                monitor.check_memory_leak(f"Function: {func.__name__}")
                return result
            finally:
                monitor.print_summary()

        return wrapper
    return decorator


@contextmanager
def memory_efficient_scope(cleanup_on_exit=True, threshold_mb=100):
    """
    Context manager for memory-efficient operations.

    Args:
        cleanup_on_exit: Whether to force cleanup on exit
        threshold_mb: Memory threshold for warnings
    """
    monitor = MemoryMonitor(threshold_mb=threshold_mb, verbose=False)

    try:
        yield monitor
    finally:
        if cleanup_on_exit:
            # Check for memory increase
            increase = monitor.get_memory_usage() - monitor.baseline_memory
            if increase > threshold_mb:
                monitor.force_cleanup()

            # Final cleanup
            if torch.cuda.is_available():
                # Only clear cache if fragmentation is high
                stats = monitor.get_memory_stats()
                if stats.get('fragmentation', 0) > 0.9:
                    torch.cuda.empty_cache()


@contextmanager
def training_step_context():
    """
    Context manager specifically for training steps.
    Ensures proper cleanup of intermediate tensors.
    """
    try:
        yield
    finally:
        # Ensure all intermediate tensors are freed
        if torch.cuda.is_available():
            torch.cuda.synchronize()

            # Check memory fragmentation
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

            # Only clear cache when fragmentation is high
            if reserved > 0 and allocated / reserved < 0.7:  # >30% fragmentation
                torch.cuda.empty_cache()


def check_tensor_memory_usage(tensor_dict, tag=""):
    """
    Check memory usage of a dictionary of tensors.

    Args:
        tensor_dict: Dictionary of tensors to check
        tag: Tag for identification
    """
    total_memory = 0
    details = []

    for name, tensor in tensor_dict.items():
        if torch.is_tensor(tensor):
            memory_mb = tensor.element_size() * tensor.nelement() / 1024**2
            total_memory += memory_mb
            details.append(f"{name}: {memory_mb:.2f} MB (shape: {tensor.shape})")

    if total_memory > 0:
        print(f"\n📦 Tensor memory usage {tag}:")
        print(f"  Total: {total_memory:.2f} MB")
        for detail in details:
            print(f"  {detail}")

    return total_memory


def profile_memory_usage(func):
    """
    Decorator to profile detailed memory usage of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_memory = torch.cuda.memory_allocated()

        try:
            result = func(*args, **kwargs)

            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            print(f"\n🔍 Memory Profile for {func.__name__}:")
            print(f"  Start: {start_memory / 1024**2:.2f} MB")
            print(f"  End: {end_memory / 1024**2:.2f} MB")
            print(f"  Peak: {peak_memory / 1024**2:.2f} MB")
            print(f"  Increase: {(end_memory - start_memory) / 1024**2:.2f} MB")

            return result

        except Exception as e:
            print(f"❌ Error in {func.__name__}: {str(e)}")
            raise

    return wrapper