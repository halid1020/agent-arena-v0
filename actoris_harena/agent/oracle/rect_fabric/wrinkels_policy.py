# Modified based on https://github.com/DanielTakeshi/gym-cloth/blob/master/examples/analytic.py#L551

import numpy as np
from scipy.spatial import distance

from agent.policies.base_policies import BasePolicy

class WrinklesPolicy(BasePolicy):
    """Approximates the wrinkle-based policy from
    https://link.springer.com/chapter/10.1007/978-3-662-43645-5_16

    Basically picks max deviation and finds the wrinkle around that
    rather than k-means + hierarchical clustering. (Their paper does
    not handle the case of cloth folded on itself.)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = kwargs['action_space']
        self.camera_height = 1.5
        
        self.flatten_threshold = 0.97
        self.no_op = [1.0, 1.0, 1.0, 1.0]

    def act(self, state, env):
        """Uses helper methods `is_point_on_top` and `get_neighbors`.

        It uses the 1D cloth array, i.e., the ground truth state.
        """
        self.camera_to_world = env.pixel_to_world_ratio
        if env.get_normalised_coverage() >= self.flatten_threshold:
            action = np.asarray(self.no_op)\
                .clip(self.action_space.low, self.action_space.high).reshape(4)
            return action
    
        def is_point_on_top(cloth, point):
            # scan down from z = height in increments of thickness until
            # the set of points with x,y in grip radius and height in
            # [z-thickness,z+thickness] is non-empty
            # all values are copied directly from cfg file, so this is not robust
            curZ = 1
            pts = []
            x = point.x
            y = point.y
            while (curZ > 0):
                for pt in cloth:
                    if (pt.x-x)*(pt.x-x) + (pt.y-y)*(pt.y-y) < 0.0002 and \
                        abs(pt.z-curZ) < 2 * 0.02:
                        pts.append(pt)
                if pts:
                    break
                    
                curZ -= 0.02
            return point in pts

        def get_neighbors(r, c, h, w):
            # helper method that gets the 3x3 matrix (or smaller if on boundary)
            # set of points surrounding the point [r, c]

            points = []
            points.append(r * w + c)
            if r > 0 and c > 0:
                points.append((r - 1) * w + c - 1)
            if r > 0:
                points.append((r - 1) * w + c)
            if c > 0:
                points.append(r * w + c - 1)
            if r < h - 1 and c < w - 1:
                points.append((r + 1) * w + c + 1)
            if r < h - 1:
                points.append((r + 1) * w + c)
            if c < h - 1:
                points.append(r * w + c + 1)
            if r > 0 and c < w - 1:
                points.append((r - 1) * w + c + 1)
            if r < h - 1 and c > 0:
                points.append((r + 1) * w + c - 1)

            return points

        # compute deviation map
        h, w = env.get_cloth_size()
        cloth = env.get_cloth()
        new_cloth = []
        for r in range(h):
            for c in range(w):
                if r%3 == 0 and c%3 == 0:
                    new_cloth.append(cloth[r * w + c])
        
        cloth = new_cloth
        h = w = int(len(new_cloth)**0.5)


        deviation = []
        # TODO
        for r in range(h):
            for c in range(w):
                if is_point_on_top(cloth, cloth[r * w + c]):
                    points = np.array([cloth[x].z for x in get_neighbors(r, c, h, w)])
                    dev = np.sum(np.abs(points - np.mean(points)))
                    deviation.append(dev)
                else:
                    deviation.append(0)
                # local_dev = [] # local_dev has depth of current point + 8 surrounding points
                # local_dev.append(cloth.pts[r * w + c].z)
                # if r > 0 and c > 0:
                #     local_dev.append(cloth.pts[(r - 1) * w + c - 1].z)
                # if r > 0:
                #     local_dev.append(cloth.pts[(r - 1) * w + c].z)
                # if c > 0:
                #     local_dev.append(cloth.pts[r * w + c - 1].z)
                # if r < h - 1 and c < w - 1:
                #     local_dev.append(cloth.pts[(r + 1) * w + c + 1].z)
                # if r < h - 1:
                #     local_dev.append(cloth.pts[(r + 1) * w + c].z)
                # if c < h - 1:
                #     local_dev.append(cloth.pts[r * w + c + 1].z)
                # if r > 0 and c < w - 1:
                #     local_dev.append(cloth.pts[(r - 1) * w + c + 1].z)
                # if r < h - 1 and c > 0:
                #     local_dev.append(cloth.pts[(r + 1) * w + c - 1].z)
                # dmean = np.mean(local_dev)
                # dev = sum([np.abs(d - dmean) for d in local_dev]) / 9.0
                # deviation.append(dev)
        # approximate largest wrinkle by choosing max deviation
        p = deviation.index(max(deviation))
        center = (cloth[p].x, cloth[p].y)
        print("CENTER:", center)
        # approximate wrinkle direction by finding the highest deviation out of neighbors
        c = p % w
        r = (p - c) // w
        indices = get_neighbors(r, c, h, w)
        indices.remove(r * w + c)
        # indices = []
        # if r > 0 and c > 0:
        #     indices.append((r - 1) * w + c - 1)
        # if r > 0:
        #     indices.append((r - 1) * w + c)
        # if c > 0:
        #     indices.append(r * w + c - 1)
        # if r < h - 1 and c < w - 1:
        #     indices.append((r + 1) * w + c + 1)
        # if r < h - 1:
        #     indices.append((r + 1) * w + c)
        # if c < h - 1:
        #     indices.append(r * w + c + 1)
        # if r > 0 and c < w - 1:
        #     indices.append((r - 1) * w + c + 1)
        # if r < h - 1 and c > 0:
        #     indices.append((r + 1) * w + c - 1)
        max_neighbor_index = max(indices, key=lambda index: deviation[index])
        wrinkle_pt = (cloth[max_neighbor_index].x, cloth[max_neighbor_index].y)
        print("SECOND WRINKLE PT:", wrinkle_pt)
        # get perpendicular angle and grab the closest edge or corner on that angle
        # approximate closest edge of cloth by finding closest cloth point to bed border
        if wrinkle_pt[0] == center[0]:
            slope = 1000 # vertical line
        else:
            slope = ((wrinkle_pt[1] - center[1]) / (wrinkle_pt[0] - center[0]))
        perp_slope = -1/slope
        # print("PERP SLOPE:", perp_slope)
        tx = 0.199

        x1, x2, y1, y2 = -tx, -tx, -tx, -tx
        if perp_slope > np.sqrt(2) + 1 or perp_slope < -(np.sqrt(2) + 1):
            # N / S direction
            # intersection with y = 1
            x1 = (tx - center[1]) / perp_slope + center[0]
            y1 = tx
            # intersection y = 0
            x2 = (-center[1]) / perp_slope + center[0]
            y2 = -tx
        elif perp_slope > 1 and perp_slope < np.sqrt(2) + 1:
            # NE / SW direction - find nearest corner
            x1, y1 = tx, tx
            x2, y2 = -tx, -tx
        elif perp_slope < np.sqrt(2) - 1 and perp_slope > -(np.sqrt(2) - 1):
            # E / W direction
            # intersection with x = 1
            y1 = perp_slope * (tx - center[0]) + center[1]
            x1 = tx
            # intersection x = 0
            y2 = perp_slope * (-center[0]) + center[1]
            x2 = -tx
        else:
            # SE / NW direction - find nearest corner
            x1, y1 = -tx, tx
            x2, y2 = tx, -tx
        closest_pt1 = min(cloth, key=lambda pt: distance.euclidean((x1, y1), (pt.x, pt.y)))
        closest_pt2 = min(cloth, key=lambda pt: distance.euclidean((x2, y2), (pt.x, pt.y)))
        if (distance.euclidean(center, (closest_pt1.x, closest_pt1.y)) < distance.euclidean(center, (closest_pt2.x, closest_pt2.y)) or
                closest_pt2.x < -tx or closest_pt2.x > tx or closest_pt2.y < -tx  or closest_pt2.y > tx): # nowhere to move
            x, y = closest_pt1.x, closest_pt1.y
            dx, dy = x1 - x, y1 - y
        else:
            x, y = closest_pt2.x, closest_pt2.y
            dx, dy = x2 - x, y2 - y
        
        
        action = (np.array([x, y, x+dx, y+dy])/(self.camera_height*self.camera_to_world))\
            .clip(self.action_space.low, self.action_space.high).reshape(4)

        return action