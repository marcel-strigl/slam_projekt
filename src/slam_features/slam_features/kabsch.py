#!/usr/bin/env python3
import math
import random

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('visual_odom')

        self.bridge = CvBridge()

        self.P_inliers = []

        # Kamera-Parameter
        self.cx = 318.525
        self.cy = 241.181
        self.fx = 526.61
        self.fy = 526.61

        #Wheel Odom
        self.real_x = 0.0
        self.real_y = 0.0
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.vektor = []

        # Vorherige Fames
        self.prev_gray = None
        self.prev_depth_img = None

        # Aktuelles Depth Frame
        self.depth_img = None

        # Geschätzte Roboter Pos
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # ORB + Matcher
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Topics Subscriben
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw',self.rgb_callback,10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw',self.depth_callback,10)
        self.subscription_wheel_odom = self.create_subscription(Odometry,'/wheel_odom/pose',self.odom_callback,10)

        # Ausgeben der Punktewolken
        self.prev_point_cloud_publisher = self.create_publisher(PointCloud2, '/orb_feature_cloud_prev', 10)
        self.current_point_cloud_publisher = self.create_publisher(PointCloud2, '/orb_feature_cloud_curr', 10)

    def odom_callback(self, msg):
        self.real_x = msg.pose.position.x
        self.real_y = msg.pose.position.y

        if self.trans_x == 0.0 and self.trans_y == 0.0:
            self.trans_x = self.real_x
            self.trans_y = self.real_y

        dx = self.real_x - self.trans_x
        dy = self.real_y - self.trans_y

        R = self.rotation_matrix_2d(math.pi / 4)

        self.vektor = np.array([dx, dy], dtype=np.float64)
        self.vektor = R @ self.vektor

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')/1000.0

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # auf depth_img Warten
        if self.depth_img is None:
            return

        # Erstes Frame speichern
        if self.prev_gray is None or self.prev_depth_img is None:
            self.prev_gray = gray.copy()
            self.prev_depth_img = self.depth_img.copy()
            return

        # Features berechnen
        kp_curr, des_curr = self.orb.detectAndCompute(gray, None)
        kp_prev, des_prev = self.orb.detectAndCompute(self.prev_gray, None)
        if des_curr is None or des_prev is None:
            return

        # Matching
        matches = self.bf.match(des_curr, des_prev)

        # Abbrechen wenn weniger als 10 matches
        if len(matches) < 10:
            self.prev_gray = gray.copy()
            self.prev_depth_img = self.depth_img.copy()
            return

        # Nur die besten 100?  Matches verwenden
        matches = sorted(matches, key=lambda x: x.distance)

        # Korrespondierende 3D-Punkte aufbauen
        prev_points_2d = []   # P vorherig
        curr_points_2d = []   # Q aktuell

        prev_points_3d = []
        curr_points_3d = []

        img_out = frame.copy()

        for m in matches:
            
            if m.distance  > 20.0: break

            kp_c = kp_curr[m.queryIdx]   # aktuelles Bild
            kp_p = kp_prev[m.trainIdx]   # vorheriges Bild

            u_c, v_c = kp_c.pt
            u_p, v_p = kp_p.pt

            u_ci, v_ci = int(round(u_c)), int(round(v_c))
            u_pi, v_pi = int(round(u_p)), int(round(v_p))


            z_curr = self.depth_img[v_ci, u_ci]
            z_prev = self.prev_depth_img[v_pi, u_pi]


            if not (0.2 <= z_curr <= 20.0 and 0.2 <= z_prev <= 20.0):
                continue

            # 3D-Punkte im Kamerakoordinatensystem
            x_curr = -z_curr * (u_c - self.cx) / self.fx
            y_curr = -z_curr * (v_c - self.cy) / self.fy
            zc_curr = z_curr

            x_prev = -z_prev * (u_p - self.cx) / self.fx
            y_prev = -z_prev * (v_p - self.cy) / self.fy
            zc_prev = z_prev

            # 2D für Kabsch: Ebene = (z, x)
            prev_points_2d.append([zc_prev, x_prev])   # P
            curr_points_2d.append([zc_curr, x_curr])   # Q

            prev_points_3d.append([zc_prev, x_prev, y_prev])######################
            curr_points_3d.append([zc_curr, x_curr, y_curr])#####################

            # Visualisierung
            cv2.circle(img_out, (u_ci, v_ci), 2, (0, 255, 0), -1)


        if len(prev_points_2d) < 4:
            self.prev_gray = gray.copy()
            self.prev_depth_img = self.depth_img.copy()
            cv2.imshow("Matches", img_out)
            cv2.waitKey(1)
            return

        prev_points_2d = np.array(prev_points_2d, dtype=np.float64)
        curr_points_2d = np.array(curr_points_2d, dtype=np.float64)
        prev_points_3d = np.array(prev_points_3d, dtype=np.float32)
        curr_points_3d = np.array(curr_points_3d, dtype=np.float32)

        # RANSAC + Kabsch
        theta_pc, t_pc, inlier_mask = self.ransac_kabsch_2d(
            prev_points_2d,
            curr_points_2d,
            iterations=120,
            threshold=0.08
        )

        if theta_pc is None:
            self.prev_gray = gray.copy()
            self.prev_depth_img = self.depth_img.copy()
            cv2.imshow("Matches", img_out)
            cv2.waitKey(1)
            return

        R_pc = self.rotation_matrix_2d(theta_pc)

        t_robot = -R_pc.T @ t_pc
        theta_robot = -theta_pc

        # Pose integrieren
        self.robot_theta += theta_robot
        c = math.cos(self.robot_theta + 3.14)
        s = math.sin(self.robot_theta + 3.14)
        R_world = np.array([[c, -s], [s, c]], dtype=np.float64)

        delta_world = R_world @ t_robot
        self.robot_x += delta_world[0]
        self.robot_y += delta_world[1]

        #Debug ausgabe
        self.get_logger().info(f"-: {self.robot_x:.2f}, y: {self.robot_y:.2f}, theta: {math.degrees(self.robot_theta):.1f}°")

        print("X: ", self.vektor[0], "      Y: ", self.vektor[1])

        prev_points_inliers_3d = prev_points_3d[inlier_mask]
        curr_points_inliers_3d = curr_points_3d[inlier_mask]

        # Punktwolken publizieren
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "kinect_depth"

        if len(prev_points_inliers_3d) > 0:
            msg_prev = point_cloud2.create_cloud_xyz32(header, curr_points_3d.tolist())
            self.prev_point_cloud_publisher.publish(msg_prev)

        if len(curr_points_inliers_3d) > 0:
            msg_curr = point_cloud2.create_cloud_xyz32(header, curr_points_inliers_3d.tolist())
            self.current_point_cloud_publisher.publish(msg_curr)

        cv2.imshow("Matches", img_out)
        cv2.waitKey(1)

        # Aktuelles Frame wird vorheriges Frame
        self.prev_gray = gray.copy()
        self.prev_depth_img = self.depth_img.copy()


    def rotation_matrix_2d(self, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def estimate_kabsch_2d(self, P, Q):
        if len(P) < 2 or len(Q) < 2:
            return None, None

        P_mean = np.mean(P, axis=0)
        Q_mean = np.mean(Q, axis=0)

        P_centered = P - P_mean
        Q_centered = Q - Q_mean

        sum1 = 0.0
        sum2 = 0.0

        for i in range(len(P)):
            p_x, p_y = P_centered[i]
            q_x, q_y = Q_centered[i]

            sum1 += q_x * p_y - q_y * p_x
            sum2 += q_x * p_x + q_y * p_y

        theta = math.atan2(sum1, sum2)
        R = self.rotation_matrix_2d(theta)
        t = P_mean - R @ Q_mean

        return theta, t


    def ransac_kabsch_2d(self, P, Q, iterations=100, threshold=0.08):
        
        best_inlier_mask = None
        best_inlier_count = 0
        
        i = list(range(len(P)))

        for n in range(iterations):
            sample_idx = random.sample(i, 2)

            P_sample = P[sample_idx]
            Q_sample = Q[sample_idx]

            theta, t = self.estimate_kabsch_2d(P_sample, Q_sample)
            if theta is None:
                continue

            R = self.rotation_matrix_2d(theta)
            errors = np.linalg.norm(P - ((R @ Q.T).T + t), axis=1)

            inlier_mask = errors < threshold
            inlier_count = int(np.sum(inlier_mask))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask

        

        if best_inlier_mask is None or best_inlier_count < 3:
            return None, None, None

        # Mit allen Inliers nochmal schätzen
        P_inliers = P[best_inlier_mask]
        Q_inliers = Q[best_inlier_mask]

        theta_refined, t_refined = self.estimate_kabsch_2d(P_inliers, Q_inliers)
        if theta_refined is None:
            return None, None, None

        return theta_refined, t_refined, best_inlier_mask


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()