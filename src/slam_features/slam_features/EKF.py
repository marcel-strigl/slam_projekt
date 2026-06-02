#!/usr/bin/env python3
import math
import random
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pcl2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

import csv

from random import *
from math import *
import matplotlib.pyplot as plt
import scipy.stats as stats


class ExtKalman:
    def __init__(self, x, state_func, meas_func, JF, JH, R, Q):
        self.x = x
        self.state_func = state_func
        self.meas_func = meas_func
        self.JF = JF
        self.JH = JH
        self.R = R
        self.Q = Q
        self.P = Q # initialize

    # set Jacobi matrix of the state transition
    def setJF(self, JF):
        self.JF = JF

    # set Jacobi Matrix of the measurement function
    def setJH(self, JH):
        self.JH = JH

    # set measurement noise -- eg. for EKF
    def setR(self, R):
        self.R = R

    # set model noise -- eg. for EKF
    def setQ(self, Q):
        self.Q = Q

    def predictState(self):
        pstate = self.state_func(self.x)
        pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
        return pstate, pP

    # return measurement prediction (\hat z_{t|t-1})
    def predictMeasurement(self):
        pmeas = self.meas_func(self.x)
        return pmeas

    # return matrix K
    def computeKalmanGain(self):
        x_tt1, P_tt1 = self.predictState()

        PHT = np.matmul(P_tt1, self.JH.transpose())         # PH^\top
        HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
        HPHTpRi = np.linalg.inv(HPHT + self.R)              # (HPH^\top + R)^{-1}
        K = np.matmul(PHT, HPHTpRi)
        return K

    # Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
    def update(self, z):
        print("State:", self.x)
        x_tt1, P_tt1 = self.predictState()
        print("Predicted state:", x_tt1)
        z_tt1 = self.predictMeasurement()
        print("Predicted measurement:", z_tt1)
        print("Actual measurement:", z)
        K = self.computeKalmanGain()
        self.x = x_tt1 + np.matmul(K, (z-z_tt1))
        self.x = self.x.flatten()
        self.P = P_tt1 - np.matmul(K, np.matmul(self.JH, P_tt1))
        return self.x, self.P


class VisualSLAM(Node):
    def __init__(self): # Konstruktor: wird nur am Start ausgeführt
        super().__init__("visual_slam")

        # Intrinsische Kamera Parameter
        self.cx = 318.525
        self.cy = 241.181
        self.fx = 526.61
        self.fy = 526.61

        # Aktuelle Pose in bezug zum Startpunkt
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.robot_vx    = 0.0
        self.robot_vy    = 0.0
        self.robot_omega = 0.0
        self.last_robot_x_for_Q = 0.0
        self.last_robot_y_for_Q = 0.0
        # Tiefenbild Initialisierung
        self.depth_img = None

        # Parameter
        self.min_matches = 10
        self.frame_skip = 1
        self.counter = 0

        # Map / Landmarken
        self.landmarks = []
        self.frame_id = 0

        # Parameter EKF
        self.dt = 0.1
        self.last_time = time.time()

        self.ekf_x = np.zeros(6, dtype=np.float64)

        self.Q_ekf = np.eye(6, dtype=np.float64) * 0.05

        self.R_ekf = np.diag([
            100.0, 
            100.0, 
            0.01 
        ])

        self.JF = np.array([
            [1, 0, self.dt, 0,       0,       0],
            [0, 1, 0,       self.dt, 0,       0],
            [0, 0, 1,       0,       0,       0],
            [0, 0, 0,       1,       0,       0],
            [0, 0, 0,       0,       1, self.dt],
            [0, 0, 0,       0,       0,       1]
        ], dtype=np.float64)

        self.JH = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ], dtype=np.float64)

        self.ekf = ExtKalman(
            self.ekf_x,
            self.ekf_state_func,
            self.ekf_meas_func,
            self.JF,
            self.JH,
            self.R_ekf,
            self.Q_ekf
        )

        self.ekf.P = np.eye(6, dtype=np.float64) * 1000.0

        # 07 Viewing Cone
        self.image_width = 640
        self.image_height = 480

        self.fov_horizontal = 2 * math.atan(self.image_width / (2 * self.fx))
        self.fov_vertical   = 2 * math.atan(self.image_height / (2 * self.fy))

        # Subscribe Imu / Odom
        self.wheel_x = 0.0
        self.wheel_y = 0.0
        self.imu_z = 0.0
        self.compare = []
        self.z = 0.0

        self.start_theta = None

        # ORB + Matcher
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.bridge = CvBridge() # Ermöglichung der Konvertierung zwischen ROS2 und OpenCV

        self.tf_broadcaster = TransformBroadcaster(self) #Transformation vom Frame "base_link" in "odom"

        self.sub_rgb = self.create_subscription(Image, "/serf01/nav_rgbd_1/rgb/image_raw",
                                                self.rgb_callback, 10)

        self.sub_depth = self.create_subscription(Image, "/serf01/nav_rgbd_1/depth/image_raw",
                                                  self.depth_callback, 10)

        self.local_map_publisher = self.create_publisher(PointCloud2, "/serf01/nav_rgbd_1/pointcloud", 10)

        self.odom_pub = self.create_publisher(Odometry, "/serf01/odometry/project_slam", 10)

        self.sub_wheel = self.create_subscription(Odometry, "/serf01/odometry/wheel", self.wheel_callback, 10)

        self.sub_imu = self.create_subscription(Odometry, "/serf01/odometry/imu", self.imu_callback, 10)

    # CALLBACKS

    def wheel_callback(self, msg):
        self.wheel_x = msg.pose.pose.position.x
        self.wheel_y = msg.pose.pose.position.y
        self.qu = msg.pose.pose.orientation

        self.z = Rotation.from_quat([self.qu.x, self.qu.y, self.qu.z, self.qu.w]).as_euler('xyz')[2]

    def imu_callback(self, msg):
        self.imu = msg.orientation

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def rgb_callback(self, msg):

        if self.start_theta == None:
            self.start_theta = self.z

        # ORB detection:
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.depth_img is None:
            return

        kp = self.orb.detect(gray, None)

        kp_valid = []   # Zwischenspeicher Variable
        for p in kp:
            u, v = int(p.pt[0]), int(p.pt[1])
            d = self.depth_img[v, u]

            if 400 < d < 4500:
                kp_valid.append(p)

        kp, des = self.orb.compute(gray, kp_valid)
        if kp is None or des is None:
            return

        # Punkte Wolke Berechnen
        pts_curr = []

        for p in kp:
            u, v = int(p.pt[0]), int(p.pt[1])
            z = float(self.depth_img[v, u])

            x = - (p.pt[0] - self.cx) * z / self.fx
            y = - (p.pt[1] - self.cy) * z / self.fy

            pts_curr.append([z, x, y])

        # Landmarken definieren und current Pointcloud in Landmarken Umwandeln
        if len(self.landmarks) == 0:
            for i in range(len(pts_curr)):
                p = pts_curr[i]

                c = math.cos(self.robot_theta)
                s = math.sin(self.robot_theta)

                gx = self.robot_x + p[0] * c - p[1] * s
                gy = self.robot_y + p[0] * s + p[1] * c

                self.landmarks.append({
                    "position": [gx, gy, p[2]],
                    "des": des[i],
                    "seen": 1,
                    "last_seen": self.frame_id
                })

        # Sichtbare Landmarken
        map_pts = []
        map_des = []
        map_idx = []

        for i, lm in enumerate(self.landmarks):
            dx = lm["position"][0] - self.robot_x
            dy = lm["position"][1] - self.robot_y

            c = math.cos(-self.robot_theta)
            s = math.sin(-self.robot_theta)

            landmark_x = dx * c - dy * s
            landmark_y = dx * s + dy * c
            landmark_z = lm["position"][2]

            # Landmarken auf Aktuelles Bild Projezieren: 3D -> 2D
            if landmark_x > 0:
                u = (-landmark_y * self.fx) / landmark_x + self.cx
                v = (-landmark_z * self.fy) / landmark_x + self.cy

                if 0 <= u <= 640 and 0 <= v <= 480:
                    map_pts.append(lm["position"][:2])
                    map_des.append(lm["des"])
                    map_idx.append(i)

        # Matching + Pose
        if self.counter == 0 and len(map_des) > 0:

            # EKF Prediction
            self.kalman_predict()

            matches = self.bf.match(np.array(map_des), des)
            matches = sorted(matches, key=lambda x: x.distance)[:50]   # nur die 50 besten matches

            if len(matches) > self.min_matches:
                P = []
                Q = []
                used_idx = set()

                for m in matches:
                    P.append(map_pts[m.queryIdx])
                    Q.append(pts_curr[m.trainIdx][:2])  # 3D -> 2D
                    used_idx.add(m.trainIdx)

                    idx = map_idx[m.queryIdx]
                    self.landmarks[idx]["seen"] += 1
                    self.landmarks[idx]["last_seen"] = self.frame_id

                R, t, theta = self.ransac(np.array(P), np.array(Q))
                print("theta_deg =", math.degrees(theta))
                print("t_x =", t[0])
                print("t_y =", t[1])
                print("robot_theta_deg =", math.degrees(self.robot_theta))

                if R is not None:

                    measured_theta_raw = theta
                    measured_theta_raw = math.atan2(
                        math.sin(measured_theta_raw),
                        math.cos(measured_theta_raw)
                    )

                    theta_step = measured_theta_raw - self.robot_theta
                    theta_step = math.atan2(
                        math.sin(theta_step),
                        math.cos(theta_step)
                    )

                    max_theta_step = math.radians(8.0)

                    theta_step = max(
                        -max_theta_step,
                        min(max_theta_step, theta_step)
                    )

                    measured_theta = self.robot_theta + theta_step
                    measured_theta = math.atan2(
                        math.sin(measured_theta),
                        math.cos(measured_theta)
                    )
                    theta_diff = measured_theta - self.robot_theta
                    theta_diff = math.atan2(
                        math.sin(theta_diff),
                        math.cos(theta_diff)
                    )

                    # Unplausiblen Kabsch-Winkel verwerfen
                    if abs(theta_diff) > math.radians(15.0):
                        measured_theta = self.robot_theta
                        theta_diff = 0.0

                    if abs(theta_diff) > math.radians(5.0):
                        measured_x = 0.85 * self.robot_x + 0.15 * t[0]
                        measured_y = 0.85 * self.robot_y + 0.15 *  t[1]
                    else:
                        measured_x = t[0]
                        measured_y = t[1]

                    self.kalman_update(measured_x, measured_y, measured_theta)
                
                    self.publish_tf()
                    self.publish_odom()

                    # neue Landmarken
                    for i in range(len(pts_curr)):
                        if i not in used_idx:
                            p = pts_curr[i]

                            c = math.cos(theta)
                            s = math.sin(theta)

                            gx = self.robot_x + p[0]*c - p[1]*s
                            gy = self.robot_y + p[0]*s + p[1]*c

                            self.landmarks.append({
                                "position": [gx, gy, p[2]],
                                "des": des[i],
                                "seen": 1,
                                "last_seen": self.frame_id
                            })

                    self.landmarks = [
                        lm for lm in self.landmarks
                        if lm["seen"] > 2 or (self.frame_id - lm["last_seen"]) < 15
                    ]
            self.counter = self.frame_skip

        # Visualisierung
        img_out = cv2.drawKeypoints(frame, kp, None, color=(0,255,0))
        cv2.imshow("Features", img_out)
        cv2.waitKey(1)

        # Pointcloud Map
        pts = [[lm["position"][0]/1000.0, lm["position"][1]/1000.0, 1 + lm["position"][2]/1000.0]
               for lm in self.landmarks]

        if len(pts) > 0:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "odom"

            self.local_map_publisher.publish(pcl2.create_cloud_xyz32(header, pts))

        self.frame_id += 1
        self.counter -= 1

    # Funktionen

    def rotation_matrix_2d(self, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def kabsch(self, P, Q):
        P_mean = np.mean(P, axis=0) # Mittelpunkt Punktewolke - Previous
        Q_mean = np.mean(Q, axis=0) # Mittelpunkt Punktewolke - Current

        # Verschiebung zur Mitte
        P_c = P - P_mean
        Q_c = Q - Q_mean

        num = np.sum(Q_c[:,0]*P_c[:,1] - Q_c[:,1]*P_c[:,0])
        den = np.sum(Q_c[:,0]*P_c[:,0] + Q_c[:,1]*P_c[:,1])

        theta = math.atan2(num, den)    # Berechnung der Drehung
        R = self.rotation_matrix_2d(theta)
        t = P_mean - R @ Q_mean         # Berechung der Translation

        return R, t, theta

    def ransac(self, P, Q, iterations=200, threshold=40):
        if len(P) < 5: # zu wenigwe Punkte = schlechte Transformation
            return None, None, None

        # Initialsiierung
        best_inliers = 0
        best_R, best_t, best_theta = None, None, None

        for _ in range(iterations):
            idx = np.random.choice(len(P), 3, replace=False) # wahl von x zufälligen korrespondierende Pkt

            R, t, theta = self.kabsch(P[idx], Q[idx])

            Q_trans = (R @ Q.T).T + t                   # Transformation von Q zu Q'
            err = np.linalg.norm(P - Q_trans, axis=1)   # Vergelich von P und Q'

            inliers = err < threshold
            count = np.sum(inliers)

            if count > best_inliers:        # bei jedem besten Inliercount - Kabsch nochmal rechnen
                best_inliers = count
                best_R, best_t, best_theta = self.kabsch(P[inliers], Q[inliers])

        return best_R, best_t, best_theta   # bester Kabsch wird zurück gegeben

    def csv_erstellung(self):
        with open("/home/marcel/SLAM_Projekt/Vergleichstabelle.csv", "w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["wheel_x", "wheel_y", "imu_thea", "visual_x", "visual_y", "visual_theta"])

            for zeile in self.compare:
                # Erstellt eine neue Liste, in der jede Zahl auf 2 Stellen und Komma formatiert ist
                formatierte_zeile = [
                    f"{wert:.2f}".replace('.', ',') if isinstance(wert, (float, int)) else wert
                    for wert in zeile
                ]
                writer.writerow(formatierte_zeile)

    # Publisher

    def publish_tf(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id  = "base_link"

        t.transform.translation.x = self.robot_x / 1000.0
        t.transform.translation.y = self.robot_y / 1000.0

        position = [t.transform.translation.x , t.transform.translation.y]

        R = self.rotation_matrix_2d(self.start_theta)

        pos = R @ position
        t.transform.translation.x  = pos[0]
        t.transform.translation.y  = pos[1]

        q = Rotation.from_euler('z', self.robot_theta + self.start_theta).as_quat()

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)



    def publish_odom(self):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id  = "base_link"

        msg.pose.pose.position.x = self.robot_x / 1000.0
        msg.pose.pose.position.y = self.robot_y / 1000.0

        position = [msg.pose.pose.position.x , msg.pose.pose.position.y]

        R = self.rotation_matrix_2d(self.start_theta)

        pos = R @ position
        msg.pose.pose.position.x  = pos[0]
        msg.pose.pose.position.y  = pos[1]

        q = Rotation.from_euler('z', self.robot_theta + self.start_theta).as_quat()

        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.pose.covariance = [0.1] * 36

        self.odom_pub.publish(msg)


        self.compare.append([self.wheel_x, self.wheel_y, self.z, msg.pose.pose.position.x , msg.pose.pose.position.y, self.robot_theta + self.start_theta])
        self.csv_erstellung()



    # EKF Funktionen
    def calculate_Q_matrix(self, delta_d_m):
        delta_d_m = max(delta_d_m, 0.001)

        # Sigma-Funktionen: 3*sigma(d) soll den Fehler abdecken
        # Werte aus Fehlerauswertung:
        # max Fehler x = 1.90 m -> sigma = 1.90 / 3 = 0.633 m
        # max Fehler y = 1.82 m -> sigma = 1.82 / 3 = 0.607 m
        # max Fehler theta = 0.48 rad -> sigma = 0.48 / 3 = 0.160 rad

        sigma_x_mm = max(633.0 * delta_d_m, 5.0)
        sigma_y_mm = max(607.0 * delta_d_m, 5.0)
        sigma_theta = max(0.160 * delta_d_m, math.radians(2.0))

        Q = np.zeros((6, 6), dtype=np.float64)

        Q[0, 0] = sigma_x_mm**2
        Q[1, 1] = sigma_y_mm**2

        # Geschwindigkeit nur leicht rauschen lassen
        Q[2, 2] = 1.0
        Q[3, 3] = 1.0

        Q[4, 4] = sigma_theta**2

        # Winkelgeschwindigkeit nur leicht rauschen lassen
        Q[5, 5] = 0.01

        return Q
    def ekf_state_func(self, x):
        x_new = np.zeros(6, dtype=np.float64)

        x_new[0] = x[0] + x[2] * self.dt
        x_new[1] = x[1] + x[3] * self.dt
        x_new[2] = x[2]
        x_new[3] = x[3]
        x_new[4] = x[4] + x[5] * self.dt
        x_new[5] = x[5]

        x_new[4] = math.atan2(math.sin(x_new[4]), math.cos(x_new[4]))

        return x_new

    def ekf_meas_func(self, x):
        return np.array([
            x[0],
            x[1],
            x[4]
        ], dtype=np.float64)

    def kalman_predict(self):
        current_time = time.time()
        self.dt = current_time - self.last_time
        self.last_time = current_time

        self.JF[0, 2] = self.dt
        self.JF[1, 3] = self.dt
        self.JF[4, 5] = self.dt

        self.ekf.setJF(self.JF)

        delta_d_mm = math.sqrt(
            (self.robot_x - self.last_robot_x_for_Q)**2 +
            (self.robot_y - self.last_robot_y_for_Q)**2
        )

        self.last_robot_x_for_Q = self.robot_x
        self.last_robot_y_for_Q = self.robot_y

        delta_d_m = delta_d_mm / 1000.0

        self.ekf.setQ(
            self.calculate_Q_matrix(delta_d_m)
        )

    def kalman_update(self, measured_x, measured_y, measured_theta):
        measured_theta = math.atan2(
            math.sin(measured_theta),
            math.cos(measured_theta)
        )

        z = np.array([
            measured_x,
            measured_y,
            measured_theta
        ], dtype=np.float64)

        x, P = self.ekf.update(z)

        self.robot_x = float(x[0])
        self.robot_y = float(x[1])

        self.robot_theta = float(x[4])
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta),
            math.cos(self.robot_theta)
        )

        self.robot_vx = float(x[2])
        self.robot_vy = float(x[3])
        self.robot_omega = float(x[5])


def main():
    rclpy.init()
    node = VisualSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()