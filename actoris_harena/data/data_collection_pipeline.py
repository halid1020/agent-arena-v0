"""
Data Collection Pipeline for Actoris Harena
===========================================
Manages thread-safe asynchronous data streams, recording sessions, and dataset metadata.
"""

import time
import threading
import numpy as np
import os
import json
from typing import Dict, Any, Optional

from actoris_harena.data.data_stream import DataStream

class DataCollectionPipeline:
    """
    Manages recording state, streams, and dataset storage with metadata tracking.

    Attributes:
        dataset_name (str): Name of the dataset directory.
        base_data_dir (str): Root directory for all datasets.
        metadata (Dict): In-memory representation of dataset metadata.
    """
    
    def __init__(self, dataset_name: str, base_data_dir: str = "./data", hardware_meta: Dict = None):
        self.streams: Dict[str, DataStream] = {}
        self.is_recording = threading.Event()
        self.should_exit = threading.Event()
        
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(base_data_dir, self.dataset_name)
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        
        self.recording_start_abs_time = 0.0
        self.current_recording_id = None
        self.current_human_name = None
        self.current_language_instruction = ""
        
        self.hardware_meta = hardware_meta or {}
        self._init_dataset()

    def _init_dataset(self):
        """Initializes directory and JSON metadata structure."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        if not os.path.exists(self.metadata_path):
            self.metadata = {
                "dataset_name": self.dataset_name,
                "hardware_configuration": self.hardware_meta,
                "streams": [],
                "recordings": {}
            }
            self._save_metadata()
        else:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def _save_metadata(self):
        """Persists the metadata dictionary to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def register_stream(self, name: str):
        """Registers a new asynchronous data stream."""
        self.streams[name] = DataStream(name)
        if name not in self.metadata["streams"]:
            self.metadata["streams"].append(name)
            self._save_metadata()

    def start_recording(self, human_name: str, language_instruction: str = ""):
        """
        Starts the local timer and enables data collection for a new episode.

        Args:
            human_name (str): A human-memorable identifier (e.g., 'fold_shirt_01').
            language_instruction (str): The global text instruction for the task.
        """
        if not self.is_recording.is_set():
            for stream in self.streams.values():
                with stream.lock:
                    stream.timestamps.clear()
                    stream.values.clear()
            
            self.recording_start_abs_time = time.time()
            self.current_recording_id = f"rec_{int(self.recording_start_abs_time)}"
            self.current_human_name = human_name
            self.current_language_instruction = language_instruction
            
            self.is_recording.set()

    def stop_and_save_recording(self):
        """Stops recording, calculates relative time, and saves episode to disk."""
        if self.is_recording.is_set():
            self.is_recording.clear()
            
            save_path = os.path.join(self.dataset_dir, f"{self.current_recording_id}.npz")
            recording_data = {}
            
            for name, stream in self.streams.items():
                with stream.lock:
                    if not stream.timestamps:
                        continue
                    abs_ts = np.array(stream.timestamps)
                    rel_ts = abs_ts - abs_ts[0] 
                    
                    recording_data[f"{name}_abs_ts"] = abs_ts
                    recording_data[f"{name}_rel_ts"] = rel_ts
                    
                    if "camera" in name:
                        recording_data[f"{name}_values"] = np.array(stream.values, dtype=object)
                    else:
                        recording_data[f"{name}_values"] = np.array(stream.values)

            np.savez_compressed(save_path, **recording_data)
            
            # Save metadata including annotations
            self.metadata["recordings"][self.current_recording_id] = {
                "human_name": self.current_human_name,
                "file": f"{self.current_recording_id}.npz",
                "start_abs_time": self.recording_start_abs_time,
                "language_instruction": self.current_language_instruction,
                "subtasks": [] # List of dicts: {"start_idx": int, "end_idx": int, "instruction": str}
            }
            self._save_metadata()

    def delete_recording(self, recording_id: str):
        """Deletes a recording and removes it from metadata."""
        if recording_id in self.metadata["recordings"]:
            file_path = os.path.join(self.dataset_dir, self.metadata["recordings"][recording_id]["file"])
            if os.path.exists(file_path):
                os.remove(file_path)
            del self.metadata["recordings"][recording_id]
            self._save_metadata()

    def push_to_stream(self, name: str, value: Any):
        if self.is_recording.is_set() and name in self.streams:
            self.streams[name].push(value)