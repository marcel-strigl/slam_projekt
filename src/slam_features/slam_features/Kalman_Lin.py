#!/usr/bin/env python3
import math
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


class VisualSLAM(Node):
    def __init__(self):
        super().__init__("visual_slam")

        self.cx = 318.525
        self.cy = 241.181
        self.fx = 526.61
        self.fy = 526.61

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        self.depth_img = None

        self.min_matches = 10
        self.frame_skip = 1
        self.counter = 0

        self.landmarks = []
        self.frame_id = 0

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ], dtype=np.float64)

        self.dt = 0.1
        self.last_time = time.time()

        self.kf_x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1000.0
        self.Q = np.eye(6, dtype=np.float64) * 0.05
        self.R_kf = np.eye(3, dtype=np.float64) * 50.0
        self.I = np.eye(6, dtype=np.float64)

        self.F = np.array([
            [1, 0, self.dt, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, self.dt],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)

        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sub_rgb = self.create_subscription(
            Image,
            "/serf01/nav_rgbd_1/rgb/image_raw",
            self.rgb_callback,
            10
        )

        self.sub_depth = self.create_subscription(
            Image,
            "/serf01/nav_rgbd_1/depth/image_raw",
            self.depth_callback,
            10
        )

        self.local_map_publisher = self.create_publisher(
            PointCloud2,
            "/serf01/nav_rgbd_1/pointcloud",
            10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/serf01/odometry/project_slam",
            10
        )

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.depth_img is None:
            return

        kp = self.orb.detect(gray, None)

        kp_valid = []
        for p in kp:
            u, v = int(p.pt[0]), int(p.pt[1])

            if u < 0 or v < 0 or v >= self.depth_img.shape[0] or u >= self.depth_img.shape[1]:
                continue

            d = self.depth_img[v, u]

            if 300 < d < 4000:
                kp_valid.append(p)

            if d==0 or np.isnan(d):
                continue

        kp, des = self.orb.compute(gray, kp_valid)

        if kp is None or des is None:
            return

        pts_curr = []

        for p in kp:
            u, v = int(p.pt[0]), int(p.pt[1])
            z = float(self.depth_img[v, u])

            x = - (p.pt[0] - self.cx) * z / self.fx
            y = - (p.pt[1] - self.cy) * z / self.fy

            pts_curr.append([z, x, y])

        if len(self.landmarks) == 0:
            for i in range(len(pts_curr)):
                self.landmarks.append({
                    "position": pts_curr[i],
                    "des": des[i],
                    "seen": 1,
                    "last_seen": self.frame_id
                })

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

            if landmark_x > 0:
                u = (-landmark_y * self.fx) / landmark_x + self.cx
                v = (-landmark_z * self.fy) / landmark_x + self.cy

                if 0 <= u <= 640 and 0 <= v <= 480:
                    map_pts.append(lm["position"][:2])
                    map_des.append(lm["des"])
                    map_idx.append(i)

        if self.counter == 0 and len(map_des) > 0:
            self.kalman_predict()

            matches = self.bf.match(np.array(map_des), des)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            if len(matches) > self.min_matches:
                P = []
                Q = []
                used_idx = set()

                for m in matches:
                    P.append(map_pts[m.queryIdx])
                    Q.append(pts_curr[m.trainIdx][:2])
                    used_idx.add(m.trainIdx)

                    idx = map_idx[m.queryIdx]
                    self.landmarks[idx]["seen"] += 1
                    self.landmarks[idx]["last_seen"] = self.frame_id

                R, t, theta, inliers, median_error = self.ransac(
                    np.array(P),
                    np.array(Q)
                )

                measurement_ok = False

                if R is not None:
                    
                    measured_x = t[0]
                    measured_y = t[1]
                    measured_theta = theta

                    measured_theta = math.atan2(
                        math.sin(measured_theta),
                        math.cos(measured_theta)
                    )

                    self.kalman_update(
                        measured_x,
                        measured_y,
                        measured_theta
                    )

                    self.publish_tf()
                    self.publish_odom()

                    for i in range(len(pts_curr)):
                        if i not in used_idx:
                            p = pts_curr[i]
                            c = math.cos(theta)
                            s = math.sin(theta)

                            gx = self.robot_x + p[0] * c - p[1] * s
                            gy = self.robot_y + p[0] * s + p[1] * c

                            self.landmarks.append({
                                "position": [gx, gy, p[2]],"des": des[i],"seen": 1,"last_seen": self.frame_id})

                    self.landmarks = [
                        lm for lm in self.landmarks
                        if lm["seen"] > 2 or (self.frame_id - lm["last_seen"]) < 30
                    ]

            self.counter = self.frame_skip

        img_out = cv2.drawKeypoints(frame, kp, None, color=(255, 0, 0))
        cv2.imshow("Features", img_out)
        cv2.waitKey(1)

        pts = [
            [
                lm["position"][0] / 1000.0,
                lm["position"][1] / 1000.0,
                1 + lm["position"][2] / 1000.0
            ]
            for lm in self.landmarks
        ]

        if len(pts) > 0:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "odom"

            self.local_map_publisher.publish(
                pcl2.create_cloud_xyz32(header, pts)
            )

        self.frame_id += 1
        self.counter -= 1

    def rotation_matrix_2d(self, theta):
        c = math.cos(theta)
        s = math.sin(theta)

        return np.array([
            [c, -s],
            [s, c]
        ], dtype=np.float64)

    def kabsch(self, P, Q):
        P_mean = np.mean(P, axis=0)
        Q_mean = np.mean(Q, axis=0)

        P_c = P - P_mean
        Q_c = Q - Q_mean

        num = np.sum(Q_c[:, 0] * P_c[:, 1] - Q_c[:, 1] * P_c[:, 0])
        den = np.sum(Q_c[:, 0] * P_c[:, 0] + Q_c[:, 1] * P_c[:, 1])

        theta = math.atan2(num, den)
        R = self.rotation_matrix_2d(theta)
        t = P_mean - R @ Q_mean

        return R, t, theta

    def ransac(self, P, Q, iterations=300, threshold=40):
        if len(P) < 8:
            return None, None, None, 0, None

        best_inliers = 0
        best_R = None
        best_t = None
        best_theta = None
        best_error = None

        for _ in range(iterations):
            idx = np.random.choice(len(P), 3, replace=False)

            R, t, theta = self.kabsch(P[idx], Q[idx])

            Q_trans = (R @ Q.T).T + t
            err = np.linalg.norm(P - Q_trans, axis=1)

            inliers = err < threshold
            count = np.sum(inliers)

            if count > best_inliers and count >= 8:
                R_refined, t_refined, theta_refined = self.kabsch(
                    P[inliers],
                    Q[inliers]
                )

                Q_refined = (R_refined @ Q.T).T + t_refined
                err_refined = np.linalg.norm(P - Q_refined, axis=1)

                best_inliers = count
                best_R = R_refined
                best_t = t_refined
                best_theta = theta_refined
                best_error = np.median(err_refined[inliers])

        return best_R, best_t, best_theta, best_inliers, best_error

    def publish_tf(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = self.robot_x / 1000.0
        t.transform.translation.y = self.robot_y / 1000.0
        t.transform.translation.z = 0.0

        q = Rotation.from_euler("z", self.robot_theta).as_quat()

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_odom(self):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = self.robot_x / 1000.0
        msg.pose.pose.position.y = self.robot_y / 1000.0
        msg.pose.pose.position.z = 0.0

        q = Rotation.from_euler("z", self.robot_theta).as_quat()

        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.pose.covariance = [0.1] * 36

        self.odom_pub.publish(msg)

    def kalman_update(self, measured_x, measured_y, measured_theta):
        z = np.array([
            [measured_x],
            [measured_y],
            [measured_theta]
        ], dtype=np.float64)

        K = self.P @ self.H.T @ np.linalg.inv(
            self.H @ self.P @ self.H.T + self.R_kf
        )

        self.kf_x = self.kf_x + K @ (z - (self.H @ self.kf_x))
        self.P = (self.I - K @ self.H) @ self.P

        self.robot_x = float(self.kf_x[0, 0])
        self.robot_y = float(self.kf_x[1, 0])

        self.robot_theta = float(self.kf_x[4, 0])
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta),
            math.cos(self.robot_theta)
        )

        self.robot_vx = float(self.kf_x[2, 0])
        self.robot_vy = float(self.kf_x[3, 0])
        self.robot_omega = float(self.kf_x[5, 0])

    def kalman_predict(self):
        current_time = time.time()
        self.dt = current_time - self.last_time
        self.last_time = current_time

        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.F[4, 5] = self.dt

        self.kf_x = self.F @ self.kf_x
        self.P = self.F @ self.P @ self.F.T + self.Q


def main():
    rclpy.init()
    node = VisualSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()