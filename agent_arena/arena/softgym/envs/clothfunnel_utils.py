import numpy as np
import pyflex
import colorsys
import cv2

from copy import deepcopy

from scipy.ndimage import rotate, shift
from skimage.measure import label, regionprops
from scipy.spatial.distance import directed_hausdorff

def load_cloth(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n) for n in line.replace('v ','').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ','').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)
    
    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices), np.array(triangle_faces), np.array(list(stretch_edges)), np.array(list(bend_edges)), np.array(list(shear_edges))


def get_default_config(
        particle_radius=0.0175,
        cloth_stiffness = (0.75, .02, .02),
        scale=0.8,
        ):
    config = {
        'scale':scale,
        'cloth_pos': [0.0, 1.0, 0.0],
        'cloth_size': [int(0.6 / particle_radius),
                       int(0.368 / particle_radius)],
        'cloth_stiff': cloth_stiffness,  # Stretch, Bend and Shear
        'camera_name': 'default_camera',
        'camera_params': {
            'default_camera':
                {
                    'render_type': ['cloth'],
                    'cam_position': [0, 2.0, 0],
                    'cam_angle': [0, -90 / 180. * np.pi, 0.], #[np.pi/2, -np.pi / 2, 0],
                    'cam_size': [480, 480],
                    'cam_fov': 39.5978 / 180 * np.pi
                }
            },
        'scene_config': {
            'scene_id': 2,
            'radius': particle_radius * scale,
            'buoyancy': 0,
            'numExtraParticles': 20000,
            'collisionDistance': 0.0006,
            'msaaSamples': 0,
        },
        'flip_mesh': 0,
        #"picker_initial_pos": np.asarray([[0.2, 0.2, 0.2], [-0.2, 0.2, 0.2]]),
    }

    return config





def set_scene(config,
              state=None,
              render_mode='cloth',
              step_sim_fn=lambda: pyflex.step(),
              ):
    if render_mode == 'particle':
        render_mode = 1
    elif render_mode == 'cloth':
        render_mode = 2
    elif render_mode == 'both':
        render_mode = 3
    
    env_idx = 0 if 'env_idx' not in config else config['env_idx']

    # print('scene_id:', config['scene_config']['scene_id'])
    # print('scene_config:', config['scene_config'])

    pyflex.set_scene_from_dict(config['scene_config']['scene_id'], config['scene_config'])

    pyflex.add_cloth_mesh(
        position=config['cloth_pos'], 
        verts=config['mesh_verts'], 
        faces=config['mesh_faces'], 
        stretch_edges=config['mesh_stretch_edges'], 
        bend_edges=config['mesh_bend_edges'], 
        shear_edges=config['mesh_shear_edges'], 
        stiffness=config['cloth_stiff'], 
        uvs=config['mesh_nocs_verts'],
        mass=config['cloth_mass'])


    random_state = np.random.RandomState(np.abs(int(np.sum(config['mesh_verts']))))
    hsv_color = [
        random_state.uniform(0.0, 1.0),
        random_state.uniform(0.0, 1.0),
        random_state.uniform(0.0, 1.0)
    ]

    rgb_color = colorsys.hsv_to_rgb(*hsv_color)
    #print('CLOTH COLOR:', rgb_color)
    pyflex.change_cloth_color(rgb_color)
    #print('Change cloth color')

    pyflex.set_camera_params_v2(config['camera_params'][config['camera_name']])

    #print('Set camera params')
    step_sim_fn()
    #print('Step sim')

    if state is not None:
        pyflex.set_positions(state['particle_pos'])
        pyflex.set_velocities(state['particle_vel'])
        pyflex.set_shape_states(state['shape_pos'])
        pyflex.set_phases(state['phase'])
    
    # particle_pos = config['init_particle_pos'].reshape(-1, 4)
    # particle_pos[:, 0] = -particle_pos[:, 0]
    # pyflex.set_positions(particle_pos)

    
    return deepcopy(config)

def calculate_iou(mask1, mask2):
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(bool)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_max_IoU(mask1, mask2):
    """
    Calculate the maximum IoU between two binary mask images,
    allowing for rotation and translation of mask1.
    
    :param mask1: First binary mask (numpy array)
    :param mask2: Second binary mask (numpy array)
    :return: Tuple of (Maximum IoU value, Matched mask)
    """

    ## if mask is rectangular, make it square by padding
    if mask1.shape[0] > mask1.shape[1]:
        pad = (mask1.shape[0] - mask1.shape[1]) // 2
        mask1 = np.pad(mask1, ((0, 0), (pad, pad)), mode='constant')
    elif mask1.shape[1] > mask1.shape[0]:
        pad = (mask1.shape[1] - mask1.shape[0]) // 2
        mask1 = np.pad(mask1, ((pad, pad), (0, 0)), mode='constant')
    
    if mask2.shape[0] > mask2.shape[1]:
        pad = (mask2.shape[0] - mask2.shape[1]) // 2
        mask2 = np.pad(mask2, ((0, 0), (pad, pad)), mode='constant')
    elif mask2.shape[1] > mask2.shape[0]:
        pad = (mask2.shape[1] - mask2.shape[0]) // 2
        mask2 = np.pad(mask2, ((pad, pad), (0, 0)), mode='constant')
    
    # print('mask1 shape:', mask1.shape)
    # print('mask2 shape:', mask2.shape)

    # if resolution above 128, we need to resize the mask to 128
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(np.uint8)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(np.uint8)
    
    

    # Get the properties of mask2
    props = regionprops(label(mask2))[0]
    center_y, center_x = props.centroid

    max_iou = 0
    best_mask = None
    
    # Define rotation angles to try
    angles = range(0, 360, 1)  # Rotate from 0 to 350 degrees in 10-degree steps
    
    for angle in angles:
        # Rotate mask1
        rotated_mask = rotate(mask1, angle, reshape=False)

        # if the mask is blank, skip
        if np.sum(rotated_mask) == 0:
            continue
        
        # Get properties of rotated mask
        rotated_props = regionprops(label(rotated_mask))[0]
        rotated_center_y, rotated_center_x = rotated_props.centroid
        
        # Calculate translation
        dy = center_y - rotated_center_y
        dx = center_x - rotated_center_x
        
        # Translate rotated mask
        translated_mask = shift(rotated_mask, (dy, dx))
        
        #translated_mask = (translated_mask > 0.1).astype(int)
        # Calculate IoU
        iou = calculate_iou(translated_mask, mask2)
        
        # Update max_iou and best_mask if necessary
        if iou > max_iou:
            max_iou = iou
            best_mask = translated_mask

    # Ensure the best_mask is binary
    if best_mask is None:
        return 0, None
    
    best_mask = (best_mask > 0.5).astype(int)

    return max_iou, best_mask



def calculate_hausdorff(mask1, mask2):
    # Get coordinates of non-zero pixels
    coords1 = np.array(np.where(mask1)).T
    coords2 = np.array(np.where(mask2)).T
    
    # Calculate directed Hausdorff distances
    forward_hausdorff = directed_hausdorff(coords1, coords2)[0]
    backward_hausdorff = directed_hausdorff(coords2, coords1)[0]
    
    # Return the max of the two directed Hausdorff distances
    return max(forward_hausdorff, backward_hausdorff)

def get_normalised_hausdorff_distance(mask1, mask2):
    """
    Calculate the minimum normalized Hausdorff distance between two binary mask images,
    allowing for rotation and translation of mask1.
    
    :param mask1: First binary mask (numpy array)
    :param mask2: Second binary mask (numpy array)
    :return: Tuple of (Minimum normalized Hausdorff distance, Matched mask)
    """
    
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(np.uint8)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(np.uint8)
    

    # Get the properties of mask2
    props = regionprops(label(mask2))[0]
    center_y, center_x = props.centroid

    min_hausdorff = np.inf
    best_mask = None
    
    # Define rotation angles to try
    angles = range(0, 360, 5)  # Rotate from 0 to 350 degrees in 10-degree steps
    
    for angle in angles:
        # Rotate mask1
        rotated_mask = rotate(mask1, angle, reshape=False)

        # if the mask is blank, skip
        if np.sum(rotated_mask) == 0:
            continue
        
        # Get properties of rotated mask
        rotated_props = regionprops(label(rotated_mask))[0]
        rotated_center_y, rotated_center_x = rotated_props.centroid
        
        # Calculate translation
        dy = center_y - rotated_center_y
        dx = center_x - rotated_center_x
        
        # Translate rotated mask
        translated_mask = shift(rotated_mask, (dy, dx))
        
        # Calculate Hausdorff distance
        hausdorff = calculate_hausdorff(translated_mask, mask2)
        
        # Update min_hausdorff and best_mask if necessary
        if hausdorff < min_hausdorff:
            min_hausdorff = hausdorff
            best_mask = translated_mask

    # Normalize by the diagonal length of the image
    height, width = mask2.shape
    diagonal_length = np.sqrt(width**2 + height**2)
    
    normalized_hausdorff = min_hausdorff / diagonal_length if diagonal_length > 0 else 0

    # Ensure the best_mask is binary
    if best_mask is None:
        return 0, None
    
    best_mask = (best_mask > 0.5).astype(int)

    return normalized_hausdorff, best_mask