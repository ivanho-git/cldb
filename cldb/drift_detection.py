"""
Workload Drift Detection for CLDB.
Monitors incoming query embeddings and performance statistics to detect concept drift,
defining workload phases to prevent unnecessary updates when the workload is stable.
"""
import numpy as np
from collections import deque
import threading
from typing import Optional, List, Dict
import time

class DriftDetector:
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.5):
        """
        Args:
            window_size: The number of recent queries to maintain in the sliding window.
            drift_threshold: Cosine distance threshold (1 - cosine_similarity) to trigger drift.
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Sliding window for current phase
        self.recent_embeddings = deque(maxlen=window_size)
        
        # The centroid of the previously established (learned) workload phase
        self.baseline_centroid: Optional[np.ndarray] = None
        
        # Tracking phases
        self.current_phase_id: int = 1
        
        self._lock = threading.Lock()
        
    def add_experience(self, embedding: np.ndarray) -> bool:
        """
        Process a new query embedding and check for drift.
        Returns True if a drift is detected, False otherwise.
        """
        with self._lock:
            # Flatten embedding if it's batched
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
                
            self.recent_embeddings.append(embedding)
            
            # Not enough data to compute drift yet
            if len(self.recent_embeddings) < self.window_size // 2:
                return False
                
            # If we don't have a baseline yet, the first full window becomes the baseline
            if self.baseline_centroid is None:
                if len(self.recent_embeddings) == self.window_size:
                    self.baseline_centroid = np.mean(list(self.recent_embeddings), axis=0)
                    # Normalize the baseline centroid
                    norm = np.linalg.norm(self.baseline_centroid)
                    if norm > 0:
                        self.baseline_centroid /= norm
                return False
                
            # Calculate the centroid of the current sliding window
            current_centroid = np.mean(list(self.recent_embeddings), axis=0)
            norm = np.linalg.norm(current_centroid)
            if norm > 0:
                current_centroid /= norm
                
            # Calculate Cosine Distance
            # Distance = 1 - cosine_similarity
            cos_sim = np.dot(current_centroid, self.baseline_centroid)
            distance = 1.0 - cos_sim
            
            if distance > self.drift_threshold:
                # Drift detected!
                self.current_phase_id += 1
                
                # Establish new baseline
                self.baseline_centroid = current_centroid
                self.recent_embeddings.clear()
                
                return True
                
            return False
            
    def get_current_phase(self) -> str:
        with self._lock:
            return f"phase_{self.current_phase_id}"
            
    def force_new_phase(self):
        """Forces the detector to step into a new phase (useful for manual task boundaries in experiments)."""
        with self._lock:
            self.current_phase_id += 1
            if len(self.recent_embeddings) > 0:
                self.baseline_centroid = np.mean(list(self.recent_embeddings), axis=0)
                norm = np.linalg.norm(self.baseline_centroid)
                if norm > 0:
                    self.baseline_centroid /= norm
            else:
                self.baseline_centroid = None
            self.recent_embeddings.clear()
            return self.get_current_phase()
