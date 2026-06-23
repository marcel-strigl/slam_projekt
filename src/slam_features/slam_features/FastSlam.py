from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math
import cv2
from .Geometry_utils import GeometryUtils
from .Configuration import Configurations, State, Coordinate
from .EKF import MapManager

# a single FastSLAM particle: ownes an independent pose estimate and EKF landmark map
class Robot():
    def __init__(self, config: Configurations, geometry: GeometryUtils):
        self.config = config
        self.geometry = geometry

        self.map_manager = MapManager(self.config)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.pose = State(x=0.0, y=0.0, theta=0.0)



    def update_robot(self, kp_clean, des_clean, depth_frame, frame_index, base_to_kinect_matrix, local_robot_pts_3d):

        delta_R = None
        delta_t = None
        delta_theta = None
        log_weight = 0.0
        pose_updated = False

        # Initial map creation
        if self.map_manager.is_empty():
            self.map_manager.initialize_map(local_robot_pts_3d, des_clean, frame_index)
 
        # Viewing Cone
        visible_des, visible_pts_glob_2d, visible_map_indices = self.geometry.test_visible_landmarks(self.map_manager.landmarks, self.pose, base_to_kinect_matrix)

        # Landmarks are in Viewing Cone
        if len(visible_des) > 0:

            # Match visible landmarks with current frame
            matches = self.bf_matcher.match(np.array(visible_des), des_clean)
            
            if len(matches) > self.config.min_matches:
                P_local = []
                Q_curr  = []
                matched_curr_indices = set()

                cos_t = math.cos(-self.pose.theta)
                sin_t = math.sin(-self.pose.theta)

                for match in matches:
                    map_idx = visible_map_indices[match.queryIdx]
                    pt_glob = visible_pts_glob_2d[match.queryIdx]

                    # TF global -> local
                    dx = pt_glob[0] - self.pose.x
                    dy = pt_glob[1] - self.pose.y
                    lx =  dx * cos_t - dy * sin_t
                    ly =  dx * sin_t + dy * cos_t

                    P_local.append([lx, ly])
                    Q_curr.append(local_robot_pts_3d[match.trainIdx][:2])
                    matched_curr_indices.add(match.trainIdx)

                    lm = self.map_manager.landmarks[map_idx]

                    lm.seen_count += 1
                    lm.last_seen = frame_index

                delta_R, delta_t, delta_theta = self.geometry.ransac_improvement(np.array(P_local), np.array(Q_curr))


                if delta_R is not None:
                    
                    # TF: local robot -> odom frame
                    cos_c = math.cos(self.pose.theta)
                    sin_c = math.sin(self.pose.theta)
                    delta_tx_odom =  delta_t[0] * cos_c - delta_t[1] * sin_c
                    delta_ty_odom =  delta_t[0] * sin_c + delta_t[1] * cos_c

                    sigma_x = self.config.sigma_x
                    sigma_y = self.config.sigma_y
                    sigma_theta = self.config.sigma_theta
                    
                    epsilon_x = np.random.normal(0.0, sigma_x)
                    epsilon_y = np.random.normal(0.0, sigma_y)
                    epsilon_theta = np.random.normal(0.0, sigma_theta)


                    # update current pose with the estimated transformation
                    self.pose.x += delta_tx_odom + epsilon_x
                    self.pose.y += delta_ty_odom + epsilon_y
                    self.pose.theta += delta_theta + epsilon_theta
                    pose_updated = True
                    
                    P_array = np.array(P_local)
                    Q_array = np.array(Q_curr)
                    
                    # Check point location after ransac rotation
                    Q_transformed = (delta_R @ Q_array.T).T + delta_t
                    errors = np.linalg.norm(P_array - Q_transformed, axis=1)

                    # Use inlier only
                    inlier_matches = [(i, match) for i, match in enumerate(matches) if errors[i] < self.config.ransac_threshold]

                    # Best inliers first
                    inlier_matches.sort(key=lambda x: errors[x[0]])


                    for i, match in inlier_matches:

                        map_idx   = visible_map_indices[match.queryIdx]     # index of matched landmarks
                        train_idx = match.trainIdx                          # index of matched keypoints

                        lm    = self.map_manager.landmarks[map_idx]
                        depth = float(depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                        # EKF-update
                        kalman_result, P, log_likelihood = lm.ekf.update(
                            np.array(local_robot_pts_3d[train_idx]),
                            np.array([self.pose.x, self.pose.y, self.pose.theta]),
                            np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]),
                            depth )

                        log_weight += log_likelihood
                        lm.pt_glob = Coordinate(x=kalman_result[0], y=kalman_result[1], z=kalman_result[2])

                    # add new landmarks to the map based on the current frame and the updated pose estimation
                    self.map_manager.add_new_landmarks(local_robot_pts_3d, 
                                                       des_clean, 
                                                       matched_curr_indices, 
                                                       self.pose, frame_index )
                    
                    # every 10 Frames: remove old landmarks
                    if frame_index % 10 == 0:          # every 10 Frames
                        self.map_manager.clean_map(frame_index)
        
                    # Reject implausible pose updates
                    if (np.linalg.norm(delta_t) > self.config.ransac_max_deviation_delta or
                        abs(delta_theta) > self.config.ransac_max_deviation_theta):

                        print(f"RANSAC bad value: |Δt|={np.linalg.norm(delta_t):.1f}mm, "f"Δθ={np.degrees(delta_theta):.1f}°")
                        log_weight = self.config.partical_filter_fail_standard_error

                else:
                    print("RANSAC failed to find a valid transformation!")
                    log_weight = self.config.partical_filter_fail_standard_error
            else:
                print(f"Not enough matches found for RANSAC!")
                log_weight = self.config.partical_filter_fail_standard_error
        else:
            print("No visible landmarks to match with!")
            log_weight = self.config.partical_filter_fail_standard_error

        return pose_updated, self.pose, log_weight


    # Create an independent copy of the particle
    def clone(self):
        new_robot = Robot(self.config, self.geometry)

        # Copy particle pose
        new_robot.pose = State(self.pose.x, self.pose.y, self.pose.theta)

        # Deep-copy landmark map
        new_robot.map_manager = self.map_manager.clone()
        return new_robot
        


@dataclass
class Keyframe:
    pose: State
    descriptors: np.ndarray
    frame_index: int
 
 
@dataclass(slots=True)
class Robots:
    id: int
    pose: State
    robot: Robot
    weight: float = 0.0
    


# VisualSLAMCore: maintains a particle filter over robot poses
class VisualSLAMCore:

    def __init__(self):
        self.config = Configurations()
        self.algo = GeometryUtils()

        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000, patchSize=31)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Best Robot pose
        self.best_pose = State(x=0.0, y=0.0, theta=0.0)

        # number of particles
        self.num_robots = self.config.num_robots
        self.robots = [Robots(  id=n,
                                pose=State(x=0.0, y=0.0, theta=0.0),
                                robot=Robot(self.config, self.algo),
                                weight=0.0                          ) 

        for n in range(self.num_robots)]

        self.best_map_manager = self.robots[0].robot.map_manager

        # use multiple Threads
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Resampling variable
        self.resample_neff_ratio = self.config.resample_threshhold

        # Loop Closure variables
        self.keyframes = []
        self.keyframe_distance = 1000.0



    # calculate the distance between two poses
    def distance_between_poses(self, pose1, pose2):
        dx = pose1.x - pose2.x
        dy = pose1.y - pose2.y
        return math.sqrt(dx*dx + dy*dy)
    
    

    # compare keyframes to find closed Loop
    def compare_with_old_keyframes(self, des_clean):

        # only compare if more than two keyframes
        if len(self.keyframes) < 2:  return
        best_matches = 0
        best_keyframe = -1

        for idx, keyframe in enumerate(self.keyframes[:-5]):
            
            matches = self.bf.match(keyframe.descriptors, des_clean)
            num_matches = len(matches)

            if num_matches > best_matches:
                best_matches = num_matches
                best_keyframe = idx

            # more than 300 matches found -> Loop Closure!
            if best_matches > 300:      print("Loop candidate found!")

        print(f"Keyframe {best_keyframe}: {best_matches} matches")




    def systematic_resample(self, weights):
        N = len(weights)

        # Generate evenly spaced sampling positions
        positions = (np.arange(N) + np.random.uniform()) / N
        indices = np.zeros(N, dtype=int)

        # Compute cumulative weight distribution
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # numerische Sicherheit gegen Rundungsfehler

        # Draw particles from cumulative distribution
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1

            else:
                j += 1

        return indices




    def resample_particles(self):

        # Compute effective particle count
        weights = np.array([r.weight for r in self.robots])
        n_eff = 1.0 / np.sum(np.square(weights))
        
        threshold = self.num_robots * self.resample_neff_ratio

        # Skip resampling if particle diversity is sufficient
        if n_eff >= threshold:
            print(f"Resampling übersprungen (N_eff={n_eff:.2f} >= {threshold:.2f})")
            return

        # Select particles using systematic resampling
        indices = self.systematic_resample(weights)
        print(f"Resampling ausgelöst (N_eff={n_eff:.2f} < {threshold:.2f}), gewählte Partikel: {indices.tolist()}")

        new_robots = []
        already_used = set()

        for new_id, src_idx in enumerate(indices):
            source_particle = self.robots[src_idx]

            # Create new particle set
            if src_idx not in already_used:
                already_used.add(src_idx)
                source_particle.id = new_id
                source_particle.weight = 1.0 / self.num_robots
                new_robots.append(source_particle)
            
            # Clone particle if selected multiple times
            else:
                cloned_robot = source_particle.robot.clone()
                new_robots.append(Robots(
                    id=new_id,
                    pose=State(source_particle.pose.x, source_particle.pose.y, source_particle.pose.theta),
                    robot=cloned_robot,
                    weight=1.0 / self.num_robots
                ))

        # Replace old particle set
        self.robots = new_robots
    



    # compute the robot pose and update the map based on the current RGB and Depth frame
    def process_frame(self, frame, depth_frame, kinect_to_base_matrix, base_to_kinect_matrix, frame_index):
        
        # find the keypoints with ORB
        kp = self.orb.detect(frame, None)
        kp_clean = []

        if depth_frame is not None:

            # cycle through keypoints
            for point in kp:

                x, y = int(point.pt[0]), int(point.pt[1])
                depth = depth_frame[y, x]

                # filter out invalid depth values
                if self.config.min_depth < depth < self.config.max_depth:
                    kp_clean.append(point)
        
        else:   return False, self.best_pose, self.best_map_manager
        
        # compute the descriptors with ORB
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)

        # skip processing if no valid keypoints found
        if kp_clean is None or des_clean is None:
            return False, self.best_pose, self.best_map_manager
        

        if len(self.keyframes) == 0:
            self.keyframes.append( Keyframe(    pose=State( self.best_pose.x,
                                                            self.best_pose.y,
                                                            self.best_pose.theta),

                                                descriptors=des_clean.copy(),
                                                frame_index=frame_index))

            print("First Keyframe saved")
            print(f"Number of Keyframes: {len(self.keyframes)}")

       

        # 3D coordinates calculation
        local_robot_pts_3d = self.algo.transform_matches_2d_to_3d(kp_clean, des_clean, kinect_to_base_matrix, depth_frame)

        log_weight = []


        # update Particles
        def _update_particle(robot_wrapper):
            pose_updated, new_pose, log_l = robot_wrapper.robot.update_robot(
                kp_clean, des_clean, depth_frame,
                frame_index, base_to_kinect_matrix, local_robot_pts_3d
            )

            robot_wrapper.pose = new_pose
            return pose_updated, log_l
        

        results      = list(self._executor.map(_update_particle, self.robots))
        pose_updated = any(r[0] for r in results)   # True wenn mind. 1 Partikel updated wurde
        log_weight   = [r[1] for r in results]

        max_log_weight = max(log_weight)

        # Normalize particle weights using the Log-Sum-Exp
        for idx, robot in enumerate(self.robots):
            log_weight_maximum = log_weight[idx] - max_log_weight
            print(f"Robot ID: {robot.id}, Log_Likelihood-Maximum: {log_weight_maximum}")
            robot.weight = np.exp(log_weight_maximum)

        # Assign uniform weights if all particles have zero likelihood
        total_weight = sum(robot.weight for robot in self.robots)
        if total_weight > 0:
            for robot in self.robots:
                robot.weight /= total_weight

        else:   # all weights == 0 -> weights = 1/N
            for robot in self.robots:
                robot.weight = 1.0 / self.num_robots

        best_robot = max(self.robots, key=lambda r: r.weight)
        print(f"Best robot ID: {best_robot.id}, Max_Likelihood: {best_robot.weight}")
        
        self.best_pose  = best_robot.pose
        best_map_manager = best_robot.robot.map_manager
        self.best_map_manager = best_map_manager
        self.compare_with_old_keyframes(des_clean)

        # Particle filter resampling
        self.resample_particles()

        #Keyframe speichern:
        if len(self.keyframes) > 0:

            last_keyframe = self.keyframes[-1]
            distance = self.distance_between_poses(self.best_pose, last_keyframe.pose)

            print(distance)
            if distance > self.keyframe_distance:

                self.keyframes.append(
                    Keyframe(   pose=State( self.best_pose.x,
                                            self.best_pose.y,
                                            self.best_pose.theta),
                                descriptors=des_clean.copy(),
                                frame_index=frame_index))

                print(f"New Keyframe saved!")
                print(f"Number of Keyframes: {len(self.keyframes)}")



        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)
            
        return pose_updated, self.best_pose, best_map_manager
    


    # Cleanly shut down the particle thread pool
    def shutdown(self):
        self._executor.shutdown(wait=True)
