"""
File Lock Utility for hmp2g-dca
Simple file locking mechanism compatible with Windows
"""
import os
import time

class FileLock:
    """
    Simple file lock for cross-process synchronization (Windows compatible)
    """
    def __init__(self, lock_file_path):
        """
        Args:
            lock_file_path: Path to the lock file
        """
        self.lock_file_path = lock_file_path
        self.lock_file = None
        self.is_locked = False
        
    def __enter__(self):
        """Acquire lock"""
        max_retries = 100
        retry_delay = 0.01  # 10ms
        
        for i in range(max_retries):
            try:
                # Try to create lock file exclusively
                # If file exists, this will fail on Windows with FileExistsError
                fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.is_locked = True
                
                # Write current PID for debugging
                with open(self.lock_file_path, 'w') as f:
                    f.write(str(os.getpid()))
                
                return self
                
            except FileExistsError:
                # Lock file exists, check if it's stale
                try:
                    # Try to read the lock file
                    with open(self.lock_file_path, 'r') as f:
                        pid_str = f.read().strip()
                    
                    # Check if the process is still running
                    if pid_str.isdigit():
                        pid = int(pid_str)
                        if not self._is_process_running(pid):
                            # Process is dead, remove stale lock
                            os.remove(self.lock_file_path)
                            continue
                except:
                    pass
                
                # Lock is held by another process, wait and retry
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Last retry failed, raise exception
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file_path} after {max_retries} retries")
        
        raise TimeoutError(f"Could not acquire lock on {self.lock_file_path}")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock"""
        if self.is_locked and os.path.exists(self.lock_file_path):
            try:
                os.remove(self.lock_file_path)
            except:
                pass
        self.is_locked = False
        
    def _is_process_running(self, pid):
        """Check if a process with given PID is running"""
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            # If psutil is not available, assume process is running
            return True
