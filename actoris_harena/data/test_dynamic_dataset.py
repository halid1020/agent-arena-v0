import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import bisect
import time
from actoris_harena.data.dynamic_trajectory_dataset import DynamicTrajectoryDataset

def generate_mock_raw_data(duration_sec: float = 5.0):
    """Generates asynchronous raw data mimicking the collection pipeline."""
    print("Generating asynchronous mock data streams...")
    data = {
        'front_camera': {'timestamps': [], 'values': []},
        'joint_states': {'timestamps': [], 'values': []},
        'actions': {'timestamps': [], 'values': []}
    }
    
    start_time = time.perf_counter()
    
    # Simulate Camera at ~30Hz (JPEG compressed)
    for t in np.arange(0, duration_sec, 1.0 / 30.0):
        data['front_camera']['timestamps'].append(start_time + t)
        mock_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, compressed = cv2.imencode('.jpg', mock_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        data['front_camera']['values'].append(compressed.tobytes())
        
    # Simulate Proprioception at ~100Hz (e.g., 7-DoF arm + 1 gripper = 8 dims)
    for t in np.arange(0, duration_sec, 1.0 / 100.0):
        data['joint_states']['timestamps'].append(start_time + t)
        data['joint_states']['values'].append(np.random.randn(8).tolist())
        
    # Simulate Actions at ~50Hz (e.g., 8 dims of target joint velocities/positions)
    for t in np.arange(0, duration_sec, 1.0 / 50.0):
        data['actions']['timestamps'].append(start_time + t)
        data['actions']['values'].append(np.random.randn(8).tolist())
        
    return data

# --- Execution and Testing ---
if __name__ == "__main__":
    # 1. Generate the raw, unsynchronized data
    raw_data = generate_mock_raw_data(duration_sec=3.0)
    
    # 2. Initialize the Dataset
    # We synchronize to 50Hz and request chunks of 8 future actions for diffusion
    dataset = DynamicTrajectoryDataset(
        raw_streams_data=raw_data, 
        target_freq_hz=50.0, 
        action_horizon=8
    )
    
    print(f"\nDataset initialized. Total valid samples: {len(dataset)}")
    
    # 3. Create the DataLoader
    # num_workers > 0 allows background CPU threads to handle JPEG decoding
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    
    # 4. Test the loading speed and batch shapes
    print("\nStarting DataLoader test...")
    start_load = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        print(f"Images shape:       {batch['image'].shape}      | dtype: {batch['image'].dtype}")
        print(f"States shape:       {batch['state'].shape}         | dtype: {batch['state'].dtype}")
        print(f"Action Chunk shape: {batch['action_chunk'].shape} | dtype: {batch['action_chunk'].dtype}")
        
        # Verify normalization
        img_min, img_max = batch['image'].min().item(), batch['image'].max().item()
        print(f"Image Value Range:  [{img_min:.2f}, {img_max:.2f}] (Expected approx [-1, 1])")
        
        break # Just test the first batch
        
    end_load = time.time()
    print(f"\nTime to fetch first batch: {(end_load - start_load) * 1000:.2f} ms")