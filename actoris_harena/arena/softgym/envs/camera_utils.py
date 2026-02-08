import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation

# def get_camera_matrix(cam_pos, cam_angle, cam_size, cam_fov):
#     focal_length_v1 = cam_size[0] / 2 / np.tan(cam_fov / 2)
#     focal_length_v2 = cam_size[1] / 2 / np.tan(cam_fov / 2)


#     cam_intrinsics = np.array([[focal_length, 0, float(cam_size[1])/2],
#                                [0, focal_length, float(cam_size[0])/2],
#                                [0, 0, 1]])
#     cam_extrinsics = np.eye(4)
#     #rotation_matrix = Rotation.from_euler('xyz', [cam_angle[1], np.pi - cam_angle[0], np.pi], degrees=False).as_matrix()
#     rotation_matrix = Rotation.from_euler('xyz', cam_angle, degrees=False).as_matrix()
#     cam_extrinsics[:3, :3] = rotation_matrix
#     cam_extrinsics[:3, 3] = cam_pos

#     return cam_intrinsics, cam_extrinsics


def get_camera_matrix(cam_pos, cam_angle, cam_size, cam_fov):
    # Assuming cam_fov is vertical FOV
    print('!!!!!!!!!cam_size:', cam_size)
    focal_length_x = 1.0*cam_size[1] / 2 / np.tan(cam_fov / 2)
    focal_length_y = 1.0*cam_size[0] / 2 / np.tan(cam_fov / 2)
    c_x = 1.0*cam_size[1] / 2
    c_y = 1.0*cam_size[0] / 2

    cam_intrinsics = np.array([[focal_length_x, 0, c_x],
                               [0, focal_length_x, c_y],
                               [0, 0, 1]])
    
    cam_extrinsics = np.eye(4)
    rotation_matrix = Rotation.from_euler('xyz', cam_angle, degrees=False).as_matrix()
    cam_extrinsics[:3, :3] = rotation_matrix
    cam_extrinsics[:3, 3] = cam_pos

    return cam_intrinsics, cam_extrinsics

def camera2world(particles_camera, camera_pos_T):
    # particles_camera: (N, 3)
    # camera_pos_T: (4, 4)
    # return: (N, 3)

    # Ensure particles_camera is a numpy array
    particles_camera = np.asarray(particles_camera)
    
    #print('particles shape', particles_camera.shape)

    # Add homogeneous coordinate (1) to each particle
    particles_homogeneous = np.hstack((particles_camera, np.ones((particles_camera.shape[0], 1))))

    #print('particles shape', particles_homogeneous.shape)
    
    # Transform particles to world coordinates
    particles_world_homogeneous = np.dot(camera_pos_T, particles_homogeneous.T).T
    
    # Remove homogeneous coordinate and return
    return particles_world_homogeneous[:, :3]


def deproject_pixel_to_point(pixel, intrinsic_matrix, depth):
    # Extract intrinsic parameters
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]
    
    # Extract pixel coordinates
    u, v = pixel
    
    # Deproject to 3D point
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z])

def pixel2camera(pixel_points, depths, intrinsic):
    pixel_points = np.atleast_2d(pixel_points).astype(int)
    depths = np.atleast_1d(depths)
    
    camera_points = []
    for pixel, depth in zip(pixel_points, depths):
        camera_points.append(deproject_pixel_to_point(pixel.tolist(), intrinsic, depth))
    return np.array(camera_points)


def pixel2world(pixel_points, camera_intrinsic_matrix, camera_extrinsic_matrix, depths):
    pixel_points = np.atleast_2d(pixel_points)
    depths = np.atleast_1d(depths)
    
    camera_points = pixel2camera(pixel_points, depths, camera_intrinsic_matrix)
    base_points = camera2world(camera_points, camera_extrinsic_matrix)
    
    return base_points

def norm_pixel2world(norm_pixel_points, camera_size, camera_intrinsic_matrix, camera_extrinsic_matrix, depths):
    norm_pixel_points = np.atleast_2d(norm_pixel_points)
    norm_pixel_points[:, 0] = (norm_pixel_points[:, 0] + 1)/2 * camera_size[0]
    norm_pixel_points[:, 1] = (norm_pixel_points[:, 1] + 1)/2 * camera_size[1]
    #print('pixel_points:', norm_pixel_points)   
    return pixel2world(norm_pixel_points, camera_intrinsic_matrix, camera_extrinsic_matrix, depths)