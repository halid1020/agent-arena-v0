## Python tutorial https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
## Elanation https://dev.intelrealsense.com/docs/post-processing-filters

import pyrealsense2 as rs
import numpy as np
import cv2

# Shearing factor for x-axis
shx = -0.2
# Shearing matrix
shear_mat = np.float32([[1, shx, 0], [0, 1, 0]])


def bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22):
    """
    Perform bilinear interpolation.
    
    Parameters:
        x, y: Coordinates of the target point.
        x1, y1, x2, y2: Coordinates of the four corners.
        q11, q21, q12, q22: Values at the four corners.
        
    Returns:
        Interpolated value at the target point.
    """
    denom = (x2 - x1) * (y2 - y1)
    w11 = (x2 - x) * (y2 - y) / denom
    w21 = (x - x1) * (y2 - y) / denom
    w12 = (x2 - x) * (y - y1) / denom
    w22 = (x - x1) * (y - y1) / denom
    
    interpolated_value = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22
    return interpolated_value


def interpolate_image(height, width, corner_values):
    interpolated_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = i/height
            y = j/width
            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            q11 = corner_values[(x1, y1)]
            q21 = corner_values[(x2, y1)]
            q12 = corner_values[(x1, y2)]
            q22 = corner_values[(x2, y2)]
            interpolated_image[i, j] = bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22)
    return interpolated_image


def plane_transform(depth_data):
    # print('depth data shape:', depth_data.shape)
    # print('depth max:', np.max(depth_data), ', min:', np.min(depth_data))
    depth_data = depth_data.astype(np.float32)/1000.0
    #depth_data = depth_data[10:depth_data.shape[0] - 10, 30:depth_data.shape[1] - 10]

    H, W = depth_data.shape
    
    ## get the 4 corners of the depth data (x, y, z)
    top_left = [0, 0, depth_data[0, 0]]
    top_right = [1, 0, depth_data[-1, 0]]
    bottom_left = [0, 1, depth_data[0, -1]]
    bottom_right = [1, 1, depth_data[-1, -1]]

    ## get the average depth of the 4 corners
    average_depth = (top_left[2] + top_right[2] + bottom_left[2] + bottom_right[2])/4.0

    
    ## create a ground truth depth, where x, y has depth top_left + (1-x) * (top_right - top_left) + y * (bottom_left - top_left)
    corner_values = {(0, 0): top_left[2], (1, 0): top_right[2], (0, 1): bottom_left[2], (1, 1): bottom_right[2]}
    ground_depth = interpolate_image(H, W, corner_values)

    depth_diff = ground_depth - average_depth

    transform_depth = depth_data - depth_diff
    
    return transform_depth


    # ## calculate the plane
    # ## plane equation: Ax + By + Cz + D = 0
    # ## A = y1 (z2 - z3) + y2 (z3 - z1) + y3 (z1 - z2)
    # ## B = z1 (x2 - x3) + z2 (x3 - x1) + z3 (x1 - x2)
    # ## C = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)
    # ## D = - (x1 (y2 z3 - y3 z2) + x2 (y3 z1 - y1 z3) + x3 (y1 z2 - y2 z1))

    # sample_points = np.asarray([top_left, top_right, bottom_left, bottom_right]).astype(np.float32)

    # A = np.ones_like(sample_points).astype(np.float32)
    # A[:, 0] = sample_points[:, 0]
    # A[:, 1] = sample_points[:, 1]
    # B = sample_points[:, 2]

    # # print('A:', A)
    # # print('B:', B)

    # ## solve the linear equation Ax = B
    # normal = np.linalg.lstsq(A, B, rcond=None)[0]
    # #print('plane coefficients:', x)

    # ## get the normal vector of the plane
    # #
    # # 
    # # normal = np.asarray([x[0], x[1], 1.0])

    # ## get the rotation matrix to make the plane parallel to the xy plane
    # rotation_matrix = rotation_matrix_from_vectors(normal, [0, 0, 1])


    # ## get all the points in the depth data
    # x = np.arange(0, depth_data.shape[0])
    # y = np.arange(0, depth_data.shape[1])
    # xx, yy = np.meshgrid(x, y)
    # xx = xx.flatten().astype(np.float32)/depth_data.shape[0]
    # yy = yy.flatten().astype(np.float32)/depth_data.shape[1]
    # zz = depth_data.flatten()

    # points = np.vstack((xx, yy, zz)).T
    # ### check first 100 points
    # ### upsample the points
    # #points = points[::10, :]
    # print('points:', points[495:505])
    # trans_points = transform_points(points, rotation_matrix)
    
    # # print('x max', trans_points[:, 0].max(), 'x min',  trans_points[:, 0].min())
    # trans_points = points
    # print('trans_points:', trans_points[495:505])

    # ## get the transformation matrix of plane to make it parallel to the xy plane
    # ## first rotate the plane to th
    # new_image_size = [128, 128]
    # x_grid = np.linspace(0, 1, new_image_size[0])
    # y_grid = np.linspace(0, 1, new_image_size[1])
    # new_depth, _, _ = np.histogram2d(trans_points[:, 0], trans_points[:, 1], bins=[x_grid, y_grid], weights=points[:, 2])

    # # get statistics of the new depth data
    # print('new depth data mean:', np.mean(new_depth), ', std:', np.std(new_depth), ', min:', np.min(new_depth), ', max:', np.max(new_depth))
    # #exit()
    
    # return new_depth





if __name__ == "__main__":
    # Configure depth and color streams
    
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.depth, 840, 840, rs.format.z16, 30)
    #config.enable_stream(rs.stream.depth, 848, 480) 
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    

    colorizer = rs.colorizer()
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2) #4
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    
    spatial.set_option(rs.option.holes_fill, 1) #3
    spatial.set_option(rs.option.filter_magnitude, 1) #5
    spatial.set_option(rs.option.filter_smooth_alpha, 1) #1
    spatial.set_option(rs.option.filter_smooth_delta, 36) #50

    hole_filling = rs.hole_filling_filter()
    temporal = rs.temporal_filter()


    pipeline = rs.pipeline()
    pipeline.start(config)
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipeline.wait_for_frames()

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            color_data = np.asanyarray(frames.get_color_frame().get_data())
            # Convert images to numpy arrays
            depth_frame = frames.get_depth_frame()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            # Show images
            cv2.namedWindow('raw depth', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('raw depth', colorized_depth)
            key = cv2.waitKey(1)

            depth_frame = decimation.process(depth_frame)
            depth_frame = depth_to_disparity.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            

            depth_data = np.asanyarray(depth_frame.get_data())
            new_depth = plane_transform(depth_data)
            print('max new depth:', np.max(new_depth), 'min new depth:', np.min(new_depth))
            #print the statistics of the depth data
            #print('depth data mean:', np.mean(depth_data), ', std:', np.std(depth_data), ', min:', np.min(depth_data), ', max:', np.max(depth_data))

            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            #color_image = np.asanyarray(color_frame.get_data())

            # Stack both images horizontally
            #images = np.hstack((color_image, colorized_depth))

            ## resize the image
            colorized_depth = cv2.resize(colorized_depth, (colorized_depth.shape[1]*2, colorized_depth.shape[0]*2))
            # Show images
            cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth', colorized_depth)
            key = cv2.waitKey(1)

            # cv2.namedWindow('transformed depth', cv2.WINDOW_AUTOSIZE)
            # normalised_new_depth = (new_depth - np.min(new_depth))/(np.max(new_depth) - np.min(new_depth))
            # normalised_new_depth = cv2.resize(normalised_new_depth, (normalised_new_depth.shape[1]*3, normalised_new_depth.shape[0]*3))
            # ## put color map
            # normalised_new_depth = cv2.applyColorMap(cv2.convertScaleAbs(normalised_new_depth, alpha=255), cv2.COLORMAP_JET)
            # ## show the new depth data after colorization
            # cv2.imshow('transformed depth', normalised_new_depth)
            # key = cv2.waitKey(1)

            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
            color_data = cv2.resize(color_data, (color_data.shape[1], color_data.shape[0]))
            cv2.imshow('color', color_data)
            key = cv2.waitKey(1)

            ## shear color image
            # sheared_color = cv2.warpAffine(color_data, shear_mat, (color_data.shape[1], color_data.shape[0]))
            # cv2.namedWindow('sheared color', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('sheared color', sheared_color)
            # key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()