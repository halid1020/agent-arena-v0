import cv2
from actoris_harena.data.data_collection_pipeline import DataCollectionPipeline
import numpy as np
import time
from pynput import keyboard
import threading

def simulate_rgb_sensor(pipeline: DataCollectionPipeline, stream_name: str, freq_hz: float):
    """Simulates an RGB camera with JPEG compression for memory safety."""
    interval = 1.0 / freq_hz
    
    while not pipeline.should_exit.is_set():
        if pipeline.is_recording.is_set():
            # Mock a 224x224 RGB image (e.g., standard ResNet/ViT input)
            raw_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Compress to JPEG before storing in RAM to prevent memory leaks
            _, compressed_frame = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            pipeline.push_to_stream(stream_name, compressed_frame.tobytes())
            
        time.sleep(interval)

def simulate_proprioception(pipeline: DataCollectionPipeline, stream_name: str, freq_hz: float):
    """Simulates high-frequency robot joint states."""
    interval = 1.0 / freq_hz
    step = 0
    
    while not pipeline.should_exit.is_set():
        if pipeline.is_recording.is_set():
            pipeline.push_to_stream(stream_name, f"joint_state_{step}")
            step += 1
        time.sleep(interval)

# --- Keyboard Controller ---

def setup_keyboard_listener(pipeline: DataCollectionPipeline):
    def on_press(key):
        try:
            if key.char == 's':
                if not pipeline.is_recording.is_set():
                    print("\n[REC] Recording STARTED.")
                    pipeline.is_recording.set()
            elif key.char == 'q':
                if pipeline.is_recording.is_set():
                    print("\n[STOP] Recording STOPPED.")
                    pipeline.is_recording.clear()
                    pipeline.should_exit.set()
                    return False # Stops the listener
        except AttributeError:
            pass

    print("Controls: Press 's' to START, 'q' to STOP and QUIT.")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener

# --- Execution ---

if __name__ == "__main__":
    pipeline = DataCollectionPipeline()
    pipeline.register_stream("joint_states") 
    pipeline.register_stream("front_camera")

    # Start sensor threads immediately, but they won't save data until 's' is pressed
    threads = [
        threading.Thread(target=simulate_proprioception, args=(pipeline, "joint_states", 100)),
        threading.Thread(target=simulate_rgb_sensor, args=(pipeline, "front_camera", 30))
    ]

    for t in threads: 
        t.daemon = True
        t.start()

    # Start keyboard listener and block main thread until 'q' is pressed
    listener = setup_keyboard_listener(pipeline)
    listener.join() 

    print("Synchronizing dataset to 50Hz...")
    dataset = pipeline.sample_synchronized(freq_hz=50.0)

    if dataset:
        print(f"Total synchronized steps: {len(dataset['master_timestamps'])}")
        print(f"Sample joint state at step 10: {dataset['joint_states'][10]}")
        print(f"Sample camera shape at step 10 (decoded): {dataset['front_camera'][10].shape}")