import numpy as np
import math
from .Configuration import Configurations, Coordinate, State

class GeometryUtils:

    def __init__(self):
        self.config = Configurations()
        self.cu = self.config.cu
        self.cv = self.config.cv
        self.f = self.config.f
        self.kinect_height = self.config.kinect_height
        self.kinect_width = self.config.kinect_width
        self.min_depth = self.config.min_depth
        self.max_depth = self.config.max_depth
        self.ransac_iterations = self.config.ransac_iterations
        self.ransac_threshold = self.config.ransac_threshold



    def ransac_improvement(self, P, Q):
        max_iterations = self.ransac_iterations
        threshold = self.ransac_threshold
        best_rotation = None
        best_translation = None
        best_theta = 0
        best_inlier_count = 0

        # Not enough points -> return the TF from all points
        if(len(P) < 5):     return best_rotation, best_translation, best_theta

        for _ in range(max_iterations):
            P_second = []
            Q_second = []

            # Random Points
            indices = np.random.choice(len(P), size=3, replace=False)
            P_subset = P[indices]
            Q_subset = Q[indices]

            # Estimate the transformation using Kabsch
            R_estimated, t_estimated, theta_estimated = self.get_kabsch_2d(P_subset, Q_subset)

            # Calculate point errors
            Q_transformed = (R_estimated @ Q.T).T + t_estimated
            errors = np.linalg.norm(P - Q_transformed, axis=1)
            
            inlier_count = np.sum(errors < threshold)

            if inlier_count > best_inlier_count:
                
                for e, p, q in zip(errors, P, Q):
                    if e < threshold:
                        P_second.append(p)
                        Q_second.append(q)

                best_rotation, best_translation, best_theta = self.get_kabsch_2d(np.array(P_second), np.array(Q_second))

                best_inlier_count = inlier_count
                
        return best_rotation, best_translation, best_theta
    

    
    def get_kabsch_2d(self, P, Q):    
        # Centroids of P and Q
        P_centroid = np.mean(P, axis=0)
        Q_centroid = np.mean(Q, axis=0)

        # Centered Pointclouds  
        P_centered = P - P_centroid
        Q_centered = Q - Q_centroid

        # Calculate theta
        theta = math.atan2(sum(Q_centered[:,0]*P_centered[:,1] - Q_centered[:,1]*P_centered[:,0]), 
                           sum(Q_centered[:,0]*P_centered[:,0] + Q_centered[:,1]*P_centered[:,1]))

        # Rotation angle -> Rotation matrix
        Rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        
        # Translation vector between Pointclouds
        Translation = P_centroid - Rotation_matrix @ Q_centroid

        return Rotation_matrix, Translation, theta
    


    def transform_matches_2d_to_3d(self, kp_clean, des_clean, kinect_to_base_matrix, depth_frame):
        local_robot_pts_3d = []

        for point, des in zip(kp_clean, des_clean):

            #Depth of points
            depth = float(depth_frame[int(point.pt[1]), int(point.pt[0])])
            
            # X, Y, Z in camera frame
            x_c = (point.pt[0] - self.cu) * depth / self.f
            y_c = (point.pt[1] - self.cv) * depth / self.f
            z_c = depth

            # Transformation in Base Coord
            pt_kinect = np.array([x_c, y_c, z_c, 1.0])
            pt_base = kinect_to_base_matrix @ pt_kinect
            
            local_robot_pts_3d.append([pt_base[0], pt_base[1], pt_base[2]])

        return local_robot_pts_3d
    


    def test_visible_landmarks(self, map_landmarks, robot_pose: State, base_to_kinect_matrix):
         
        visible_des = []
        visible_pts_glob_2d = []
        visible_map_indices = []

        c = math.cos(-robot_pose.theta)
        s = math.sin(-robot_pose.theta)

        for idx, lm in enumerate(map_landmarks):

            # Relative landmark position to robot
            delta_x = lm.pt_glob.x - (robot_pose.x)
            delta_y = lm.pt_glob.y - (robot_pose.y)

            # Transform to robot coordinates
            lx = delta_x * c - delta_y * s
            ly = delta_x * s + delta_y * c
            lz = lm.pt_glob.z

            # Base -> Kinect transformation
            pt_cam = base_to_kinect_matrix.dot((lx, ly, lz, 1.0))


            c_x = pt_cam[0]
            c_y = pt_cam[1]
            c_z = pt_cam[2]
            
            # Check for valid Camera Coord
            if 0 < c_z < self.max_depth:

                # Project to 2D image plane
                u_p = (c_x * self.f) / c_z + self.cu
                v_p = (c_y * self.f) / c_z + self.cv

                # Check if projected point is within image bounds
                if 0 <= u_p <= self.kinect_width and 0 <= v_p <= self.kinect_height:
                    visible_des.append(lm.des)
                    visible_pts_glob_2d.append((lm.pt_glob.x, lm.pt_glob.y))
                    visible_map_indices.append(idx)

        return visible_des, visible_pts_glob_2d, visible_map_indices