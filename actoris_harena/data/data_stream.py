import time
import threading
import numpy as np
import bisect
from typing import Dict, Any, List, Tuple

class DataStream:
    """Thread-safe buffer for a single modality stream."""
    def __init__(self, name: str):
        self.name = name
        self.timestamps: List[float] = []
        self.values: List[Any] = []
        self.lock = threading.Lock()

    def push(self, value: Any):
        with self.lock:
            self.timestamps.append(time.perf_counter())
            self.values.append(value)

    def get_nearest(self, query_ts: float) -> Any:
        with self.lock:
            if not self.timestamps:
                return None
            
            idx = bisect.bisect_left(self.timestamps, query_ts)
            
            if idx == 0: return self.values[0]
            if idx == len(self.timestamps): return self.values[-1]
            
            before_ts = self.timestamps[idx - 1]
            after_ts = self.timestamps[idx]
            
            if (query_ts - before_ts) <= (after_ts - query_ts):
                return self.values[idx - 1]
            else:
                return self.values[idx]