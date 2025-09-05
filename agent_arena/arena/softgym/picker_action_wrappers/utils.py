import numpy as np

def pixel_to_world(p, depth, cam_intrinsics, cam_pose, cam_size):
    # Normalize pixel coordinates from [-1, 1] to [0, 1]
    p_norm = (p + 1) / 2
    
    # swap y and x
    p_norm = np.array([p_norm[1], p_norm[0]])

    #print('cam_size:', cam_size)
    # Convert to pixel coordinates
    pixel_x = p_norm[0] * cam_size[0]
    pixel_y = p_norm[1] * cam_size[1]

    # Create homogeneous pixel coordinates
    pixel_homogeneous = np.array([pixel_x, pixel_y, 1])

    # Convert to camera coordinates
    cam_coords = np.linalg.inv(cam_intrinsics) @ (depth * pixel_homogeneous)
    #print('cam_coords:', cam_coords)

    # Convert to homogeneous coordinates
    cam_coords_homogeneous = np.append(cam_coords, 1)

    # Transform to world coordinates
    world_coords = cam_pose @ cam_coords_homogeneous

    return world_coords[:3]  # Return only x, y, z