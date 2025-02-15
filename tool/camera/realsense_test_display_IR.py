# Code to display the IR image from the RealSense camera
# This code is based on the example provided by Intel RealSense SDK 2.0
# https://community.intel.com/t5/Items-with-no-label/How-to-show-d435-IR-image-using-python-wrapper-pyrealsense2/m-p/545526

import pyrealsense2 as rs
import numpy as np
import cv2
 
# We want the points object to be persistent so we can display the 
#last cloud when a frame drops
points = rs.points()
 
# Create a pipeline
pipeline = rs.pipeline()
#Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
# Start streaming
profile = pipeline.start(config)
 
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1) # Left IR Camera, it allows 1, 2 or no input
        image = np.asanyarray(ir1_frame.get_data())
        cv2.namedWindow('IR Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()