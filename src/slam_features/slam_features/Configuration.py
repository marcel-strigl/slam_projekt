#!/usr/bin/env python3

import numpy as np
from math import pi
from dataclasses import dataclass

@dataclass(slots=True)
class Coordinate:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class State:
    x: float
    y: float
    theta: float


class Configurations:
    def __init__(self):

        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181

        # Picture Size
        self.kinect_width = 640
        self.kinect_height = 480

        # Focal length
        self.f = 526.61

        # valid depth range Tiefenbereich (mm)
        self.min_depth = 400
        self.max_depth = 7500

        # RANSAC Configuration
        self.ransac_iterations = 100
        self.ransac_threshold = 40  # mm
        self.ransac_max_deviation_delta = 400  # mm
        self.ransac_max_deviation_theta = 2.35  # rad

        # max new Landmarks per frame
        self.max_new_landmarks_per_frame = 50
        self.min_matches = 30

        # Number of skipped frames after update
        self.frame_counter = 1

        # Noise
        self.sigma_x = 5          # mm
        self.sigma_y = 5          # mm
        self.sigma_theta = 0.002  # rad
        
        # Sigma R Approximation
        self.depth_noise_floor = 0.001477       
        self.depth_noise_coeff = 0.002294       
        self.lateral_error = 0.8 / 3

        # Number of Virtual Robots
        self.num_robots = 5 
        self.partical_filter_fail_standard_error = -700

        # Resampling threshhold
        self.resample_threshhold = 0.75