"""
Test Suite for Actoris Harena Data Pipeline
===========================================
Mocks a complex dual-arm hardware setup with multiple cameras and tactile sensors.
"""

import cv2
import numpy as np
import time
import threading
from torch.utils.data import DataLoader

from data_collection_pipeline import DataCollectionPipeline
from dynamic_trajectory_dataset import DynamicTrajectoryDataset

# TODO: sample different intervals for different data types. sensors like [-2, 4], actions like [-2, 8].

def simulate_sensor_stream(pipeline, stream_name, freq_hz, data_generator):
    interval = 1.0 / freq_hz
    start_time = time.time()
    
    while not pipeline.should_exit.is_set():
        if pipeline.is_recording.is_set():
            t = time.time() - start_time
            pipeline.push_to_stream(stream_name, data_generator(t))
        time.sleep(interval)

if __name__ == "__main__":
    # Hardware Metadata Configuration
    hardware_meta = {
        "robot_type": "Dual UR5e with Robotiq 2F-85",
        "camera_intrinsics": {"front": [600, 0, 320, 0, 600, 240, 0, 0, 1]},
        "tactile_sensor_type": "GelSight Mini (4 units)"
    }
    
    pipeline = DataCollectionPipeline(dataset_name="dual_arm_dataset", hardware_meta=hardware_meta)
    
    # Register Complex Topology
    streams = [
        "front_camera", "top_camera", "left_wrist_camera", "right_wrist_camera",
        "left_arm_joints", "right_arm_joints", 
        "left_tactile", "right_tactile",
        "left_actions", "right_actions"
    ]
    for s in streams: pipeline.register_stream(s)

    # Generators based on time (t) for meaningful visuals and plots
    def gen_img(t):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        # Draw a moving circle to simulate a moving end-effector
        x = int(112 + 80 * np.cos(t * 2))
        y = int(112 + 80 * np.sin(t * 2))
        cv2.circle(frame, (x, y), 20, (0, 255, 100), -1)
        _, c = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return c.tobytes()
    
    def gen_joints(t):
        # Generate smooth sine waves for the line plots
        return [np.sin(t), np.cos(t), np.sin(t*0.5), np.cos(t*0.5), 0, 0, 0]
        
    def gen_tactile(t):
        return np.random.randn(4, 16, 16).tolist() # Mock 4x16x16 tactile arrays
    
    threads = []
    # Start Cameras (30Hz)
    for cam in ["front_camera", "top_camera", "left_wrist_camera", "right_wrist_camera"]:
        threads.append(threading.Thread(target=simulate_sensor_stream, args=(pipeline, cam, 30, gen_img)))
    
    # Start States & Actions (100Hz & 50Hz)
    for arm in ["left", "right"]:
        threads.append(threading.Thread(target=simulate_sensor_stream, args=(pipeline, f"{arm}_arm_joints", 100, gen_joints)))
        threads.append(threading.Thread(target=simulate_sensor_stream, args=(pipeline, f"{arm}_tactile", 100, gen_tactile)))
        threads.append(threading.Thread(target=simulate_sensor_stream, args=(pipeline, f"{arm}_actions", 50, gen_joints)))

    for t in threads: 
        t.daemon = True
        t.start()

    print("\nRecording Episode 1...")
    pipeline.start_recording(human_name="fold_towel_01", language_instruction="Fold the blue towel in half.")
    time.sleep(3.0) # Record for 3 seconds to get a good trajectory curve
    # TODO: we also want to save the recordling length in miliseconds in dataset, but in the visualiser we want this as 
    pipeline.stop_and_save_recording()
    
    pipeline.should_exit.set()
    
    # Test DataLoader
    dataset = DynamicTrajectoryDataset("./data/dual_arm_dataset", target_freq_hz=50.0, action_horizon=2)
    dataloader = DataLoader(dataset, batch_size=2)
    
    batch = next(iter(dataloader))
    print(f"\nDataLoader Output:")
    print(f"Front Camera Shape: {batch['front_camera'].shape}")
    print(f"Left Arm State: {batch['left_arm_joints'].shape}")
    print(f"Left Action Chunk: {batch['left_actions'].shape}")