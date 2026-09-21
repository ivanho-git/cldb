"""
Adaptive Background Training Scheduler for CLDB.
Monitors the drift detection module and triggers asynchronous continual learning 
to avoid blocking query execution.
"""
import threading
import time
from typing import Optional, Callable
from cldb.logger import get_logger

class OnlineScheduler:
    def __init__(self, 
                 train_callback: Callable[[], None],
                 drift_detector=None,
                 check_interval_seconds: float = 5.0,
                 min_queries_before_train: int = 50):
        """
        Args:
            train_callback: A function that triggers `train_on_experience`.
            drift_detector: The DriftDetector instance to monitor for concept drift.
            check_interval_seconds: How often the background thread wakes up to check.
            min_queries_before_train: Minimum new queries required before a training cycle is allowed.
        """
        self.train_callback = train_callback
        self.drift_detector = drift_detector
        self.check_interval_seconds = check_interval_seconds
        self.min_queries_before_train = min_queries_before_train
        
        self.logger = get_logger("online_scheduler")
        self.queries_since_last_train = 0
        self.is_running = False
        self.is_training = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def increment_query_count(self):
        """Called by the pipeline every time a query is added to the replay buffer."""
        with self._lock:
            self.queries_since_last_train += 1
            
    def trigger_drift(self):
        """Called when the drift detector explicitly detects a phase change."""
        self.logger.info("Drift detected. Flagging scheduler for urgent training.")
        # If drift is detected, we can force a train if we aren't already
        self._try_start_training()
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info("Online Scheduler started in background.")
        
    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _run_loop(self):
        while self.is_running:
            time.sleep(self.check_interval_seconds)
            
            with self._lock:
                should_train = self.queries_since_last_train >= self.min_queries_before_train
                
            if should_train:
                self._try_start_training()
                
    def _try_start_training(self):
        with self._lock:
            if self.is_training:
                return
            self.is_training = True
            
        # Spawn training in a separate thread so the scheduler loop isn't blocked
        train_thread = threading.Thread(target=self._execute_training, daemon=True)
        train_thread.start()
        
    def _execute_training(self):
        try:
            self.logger.info("Starting background training cycle.")
            start_t = time.time()
            
            self.train_callback()
            
            duration = time.time() - start_t
            self.logger.info(f"Background training completed in {duration:.2f}s.")
            
            with self._lock:
                self.queries_since_last_train = 0
                
        except Exception as e:
            self.logger.error(f"Error during background training: {e}")
        finally:
            with self._lock:
                self.is_training = False
