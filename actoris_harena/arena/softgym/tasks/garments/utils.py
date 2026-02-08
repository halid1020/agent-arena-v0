# codes are borrow from 
# https://github.com/real-stanford/cloth-funnels/blob/main/cloth_funnels/learning/utils.py#L120
import numpy as np
import open3d as o3d
from copy import deepcopy


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    # assert centroid_A and centroid_B has no nan
    if np.isnan(centroid_A).any() or np.isnan(centroid_B).any():
        print('A', A)
        print('centroid_A', centroid_A)
        print('B', B)
        print('centroid_B', centroid_B)
    assert not np.isnan(centroid_A).any()
    assert not np.isnan(centroid_B).any()

    t = -R @ centroid_A + centroid_B

    return R, t


def superimpose(current_verts, goal_verts, indices=None, symmetric_goal=False):

    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()
    # flipped_goal_verts = goal_verts.copy()
    # flipped_goal_verts[:, 0] = 2*np.mean(flipped_goal_verts[:, 0]) - flipped_goal_verts[:, 0]

    if indices is not None:
        #assert len(indices) > 0
        #print('indices', indices)
        R, t = rigid_transform_3D(current_verts[indices].T, goal_verts[indices].T)
    else:
        R, t = rigid_transform_3D(current_verts.T, goal_verts.T)

    ## assert R and t has no nan
    assert not np.isnan(R).any()
    assert not np.isnan(t).any()

    icp_verts = (R @ current_verts.T + t).T

    return icp_verts

def transform_verts(verts, goal_verts, threshold, iteration=1):
    indices = None
    transform_verts = verts.copy()
    #print('threshold', threshold)
    for i in range(iteration):
        transform_verts = superimpose(transform_verts, goal_verts, indices=indices)
        distances = np.linalg.norm(transform_verts - goal_verts, axis=1)
        #print('min distance', np.min(distances))
        indices = distances < threshold
        if np.sum(indices) == 0:
            indices = None
            break
    
    transform_verts = superimpose(transform_verts, goal_verts, indices=indices)

    return transform_verts

def get_deform_distance(current_verts, goal_verts, threshold):
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]    
    transform_verts_ = transform_verts(
        current_verts, goal_verts, threshold, iteration=5)
    deform_distance_regular = np.mean(np.linalg.norm(transform_verts_ - goal_verts, axis=1))
    deform_distance_flipped = np.mean(np.linalg.norm(transform_verts_ - flipped_goal_verts, axis=1))
    deform_l2_distance = min(deform_distance_regular, deform_distance_flipped)
    ## assert is a number
    assert not np.isnan(deform_l2_distance)
    return deform_l2_distance

def get_rigid_distance(current_verts, goal_verts, threshold):
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, current_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T
    reverse_goal_verts = transform_verts(
        reverse_goal_verts, current_verts, threshold, iteration=1)
    
    rigid_distance_regular = np.mean(np.linalg.norm(goal_verts - reverse_goal_verts, axis=1))
    rigid_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - reverse_goal_verts, axis=1))
    rigid_distance = min(rigid_distance_regular, rigid_distance_flipped)
    ## assert is a number
    assert not np.isnan(rigid_distance)
    return rigid_distance

def deformable_distance(goal_verts, current_verts, max_coverage, 
        deformable_weight=0.65, flip_x=True, icp_steps=1000, scale=None):

    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    #flatten goals
    #print('shape of goal_verts', goal_verts.shape)
    #print('shape of current_verts', current_verts.shape)
    goal_verts[:, 2] = 0
    current_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]

    real_l2_distance = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
    real_l2_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
    if real_l2_distance_flipped < real_l2_distance:
        real_l2_distance = real_l2_distance_flipped


    #GOAL is RED
    goal_vert_cloud = o3d.geometry.PointCloud()
    goal_vert_cloud.points = o3d.utility.Vector3dVector(goal_verts.copy())
    goal_vert_cloud.paint_uniform_color([1, 0, 0])

    normal_init_vert_cloud = deepcopy(goal_vert_cloud)

    flipped_goal_vert_cloud = o3d.geometry.PointCloud()
    flipped_goal_vert_cloud.points = o3d.utility.Vector3dVector(flipped_goal_verts.copy())
    flipped_goal_vert_cloud.paint_uniform_color([0, 1, 1])

    goal_vert_cloud += flipped_goal_vert_cloud
    #CURRENT is GREEN
    verts_cloud = o3d.geometry.PointCloud()
    verts_cloud.points = o3d.utility.Vector3dVector(current_verts.copy())
    verts_cloud.paint_uniform_color([0, 1, 0])

    THRESHOLD_COEFF = 0.3
    threshold = np.sqrt(max_coverage) * THRESHOLD_COEFF
    
    #### Get Deomrable Distance
    deform_l2_distance = get_deform_distance(current_verts, goal_verts, threshold)
   
    #### Get Rigid Distance
    rigid_l2_distance = get_rigid_distance(current_verts, goal_verts, threshold)

    #make reward scale invariant
    assert(max_coverage != 0 or scale != 0)
    if scale is None:
        deform_l2_distance /= np.sqrt(max_coverage)
        rigid_l2_distance /= np.sqrt(max_coverage)
        real_l2_distance /= np.sqrt(max_coverage)
    else:
        deform_l2_distance /= scale
        rigid_l2_distance /= scale
        real_l2_distance /= scale

    weighted_distance = deformable_weight * deform_l2_distance + (1 - deformable_weight) * rigid_l2_distance

    return weighted_distance, deform_l2_distance, rigid_l2_distance, real_l2_distance