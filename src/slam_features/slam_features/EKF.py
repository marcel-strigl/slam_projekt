import math
from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Any
from .Configuration import Configurations, Coordinate, State


class EKF:
    def __init__(self, x):

        self.config = Configurations()

        # Camera Parameters
        self.cu = self.config.cu
        self.cv = self.config.cv
        self.f  = self.config.f

        self.x = x
        self.state_func = 0
        self.meas_func = 0

        # EKF Init
        self.JF = np.eye(3) # Jakobian of f
        self.Q  = np.diag([1.0, 1.0, 5.0]) # prozess  noise  covariance matrix
        self.P  = None # state covariance matrix


    # set measurement noise
    def setR(self, pt, depth_value, c, s):

        # approximate measurement noise
        s_z, s_x = self.sigma_R_approximation(depth_value)

        R_sigma_pixel = np.array([[s_x**2,    0.0,    0.0],
                                  [   0.0, s_x**2,    0.0],
                                  [   0.0,    0.0, s_z**2]])

        J_pixel = np.array([[depth_value/self.f,                0.0, (pt[0]-self.cu)/self.f],
                            [               0.0, depth_value/self.f, (pt[1]-self.cv)/self.f],
                            [               0.0,                0.0,                    1.0]])

        R_rot_kb = np.array([[  c,   s, 0.0],
                             [ -s,   c, 0.0],
                             [0.0, 0.0, 1.0]])

        R_sigma_kinect = J_pixel@R_sigma_pixel@J_pixel.T
        R_sigma_base = R_rot_kb.T@R_sigma_kinect@R_rot_kb
        self.R = R_sigma_base



    # Predict landmark state and covariance
    def predictState(self):
        predicted_state = self.x
        predicted_cov = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
        return predicted_state, predicted_cov



    # Predict landmark measurement in robot frame
    def predictMeasurement(self, rob_curr, c, s):
        predicted_measurement = np.array([c*(self.x[0]-rob_curr[0])+s*(self.x[1]-rob_curr[1]),
                          -s*(self.x[0]-rob_curr[0])+c*(self.x[1]-rob_curr[1]),
                          self.x[2]                 -rob_curr[2]])
        return predicted_measurement
    


    # Compute Kalman gain -> K-matrix
    def computeKalmanGain(self):
        PH_T = np.matmul(self.P, self.JH.transpose())             
        temp_res = np.linalg.inv(np.matmul(self.JH, PH_T) + self.R)      
        K = np.matmul(PH_T, temp_res)
        return K



    # Update function for EKF
    def update(self, z, curr_rob, pt, depth_value):

        c = cos(curr_rob[2])
        s = sin(curr_rob[2])

        self.setR(pt, depth_value, c, s)

        # first observation: init covaraince from measurement uncertainty
        if self.P is None:      
            self.P = self.R.copy()

        # Predict state and covariance
        x_tt1, P_tt1 = self.predictState()

        self.JH = np.array([[  c,   s, 0.0],
                            [ -s,   c, 0.0],
                            [0.0, 0.0, 1.0]])
        
        # Update predicted covariance
        self.P = P_tt1

        # Predict measurement
        z_tt1 = self.predictMeasurement(curr_rob, c, s)

        K = self.computeKalmanGain()

        # Update State
        self.x = self.x + np.matmul(K, (z-z_tt1))
        self.P = self.P - np.matmul(K, np.matmul(self.JH, self.P))
        
        likelihood = self.compute_measurement_likelihood(z, z_tt1)
        return self.x, self.P, likelihood
    

    
    def sigma_R_approximation(self, depth_value: float):
        depth_noise_floor = self.config.depth_noise_floor
        depth_noise_coeff = self.config.depth_noise_coeff

        depth_m = depth_value / 1000.0 

        # depth noise
        depth_noise = (depth_noise_floor + depth_noise_coeff * (depth_m - 0.4)**2) * 1000

        # lateral error
        lateral_error = self.config.lateral_error

        return depth_noise, lateral_error



    # Compute measurement log-likelihood
    def compute_measurement_likelihood(self, z, z_tt1):

        # Compute measurement residual
        innovation = (z - z_tt1)

        try:
            PHT = np.matmul(self.P, self.JH.transpose())        
            HPHT = np.matmul(self.JH, PHT)       

            # Compute innovation covariance
            S = HPHT + self.R  # Innovation covariance
            S_inv = np.linalg.inv(S)
            S_det = np.linalg.det(S)

            # Compute Mahalanobis distance term
            exponent = -0.5 * innovation.T @ S_inv @ innovation

            # Compute Gaussian log-likelihood
            log_likelihood = -0.5 * (3 * np.log(2 * np.pi) + np.log(S_det)) + exponent

            return log_likelihood
        
        except np.linalg.LinAlgError: # Assign very low likelihood if covariance is singular
            return self.config.partical_filter_fail_standard_error



    # Clone EKF state for particle resampling
    def clone(self):
        # Copy state vector
        cloned = EKF(self.x.copy())

        # Share read-only configuration
        cloned.config = self.config

        # Copy covariance matrix
        cloned.P = self.P.copy() if self.P is not None else None

        # Copy state transition Jacobian
        cloned.JF = self.JF.copy()

        return cloned



@dataclass(slots=True)
class Landmark:
    pt_glob: Coordinate
    des: Any
    seen_count: int
    last_seen: int
    ekf: EKF


# store Landmarks in 3D
class MapManager:
    def __init__(self, config):
        self.config = config        
        self.landmarks = []
        


    # Check if Map is empty
    def is_empty(self):
        return len(self.landmarks) == 0
    


    # create initial map with the first frames keypoints
    def initialize_map(self, local_pts_3d, descriptors, frame_index):
        for i in range(len(local_pts_3d)):
            
            pt = local_pts_3d[i]
            coordinate = Coordinate(pt[0], pt[1], pt[2])

            self.landmarks.append(Landmark( pt_glob=coordinate,
                                            des=descriptors[i],
                                            seen_count=1,
                                            last_seen=frame_index,
                                            ekf=EKF(np.array(pt))))
            


    # add new, unmatched points to the global map
    def add_new_landmarks(self, local_pts_3d, descriptors, matched_curr_indices, curr_pose: State, frame_index):
        curr_pos_x, curr_pos_y, curr_theta = curr_pose.x, curr_pose.y, curr_pose.theta
        added_landmarks = 0

        for i in range(len(local_pts_3d)):

            # add only a given number of landmarks
            if added_landmarks >= self.config.max_new_landmarks_per_frame:
                break

            # only landmarks that are not in the Map
            if i not in matched_curr_indices:
                pt = local_pts_3d[i]

                c = math.cos(curr_theta)
                s = math.sin(curr_theta)

                gx = curr_pos_x + pt[0] * c - pt[1] * s
                gy = curr_pos_y + pt[0] * s + pt[1] * c

                coordinate = Coordinate(gx, gy, pt[2])

                self.landmarks.append(Landmark( pt_glob=coordinate,
                                                des=descriptors[i],
                                                seen_count=1,
                                                last_seen=frame_index,
                                                ekf=EKF(np.array([gx, gy, pt[2]]))))

                added_landmarks += 1



    # delete all landmarks with high last_seen count
    def clean_map(self, frame_index):
        self.landmarks = [lm for lm in self.landmarks if (frame_index - lm.last_seen) < 75]



    # get all landmark points in the format for PointCloud2 message
    def get_all_points_for_msg(self):
        return [[lm.pt_glob.x/1000.0, lm.pt_glob.y/1000.0, lm.pt_glob.z/1000.0] for lm in self.landmarks]




    # Clone landmark map for particle resampling
    def clone(self):
        new_manager = MapManager(self.config)

        # Copy landmark state and EKF
        new_manager.landmarks = [
            Landmark(
                pt_glob=Coordinate(lm.pt_glob.x, lm.pt_glob.y, lm.pt_glob.z),
                des=lm.des,
                seen_count=lm.seen_count,
                last_seen=lm.last_seen,
                ekf=lm.ekf.clone())
            for lm in self.landmarks]
        
        return new_manager