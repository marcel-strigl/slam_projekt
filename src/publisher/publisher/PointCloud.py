#!/usr/bin/env python3
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('matcher_inkl_angels')

        self.bridge = CvBridge()

        # Kamera-Intrinsics
        self.cx = 318.525
        self.cy = 241.181
        self.fx = 526.61
        self.fy = 526.61

        # Vorheriges RGB-Bild
        self.prev_gray = None

        # Letztes Tiefenbild
        self.prev_depth_img = None

        # ORB + Matcher
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # RGB Subscriber
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw',self.rgb_callback,10)

        # Depth Subscriber
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw',self.depth_callback,10)

        # PointCloud Publisher
        self.pc_pub = self.create_publisher(PointCloud2,'/orb_matched_feature_cloud',10)

    def depth_callback(self, msg):
        self.prev_depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Erstes Bild nur speichern
        if self.prev_gray is None:
            self.prev_gray = gray
            return

        # ORB Features
        kp1, des1 = self.orb.detectAndCompute(gray, None)
        kp2, des2 = self.orb.detectAndCompute(self.prev_gray, None)

        # Matching
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        h, w = self.prev_depth_img.shape[:2]

        # gültige Punkte vor dem Zusammenfassen: (u, v, z)
        valid_features = []

        for m in matches:
            kp = kp1[m.queryIdx]   # Punkt im aktuellen Bild
            u, v = kp.pt
            u_i, v_i = int(round(u)), int(round(v))

            if not (0 <= u_i < w and 0 <= v_i < h):
                continue

            depth_raw = self.prev_depth_img[v_i, u_i]

            if not self.is_valid_depth(depth_raw):
                continue

            z = self.depth_raw/ 1000.0

            # sinnvoller Tiefenbereich
            if z < 0.0 or z > 4.0:
                continue

            # lokale Tiefenstabilität
            if not self.is_depth_locally_stable(u_i, v_i):
                continue

            valid_features.append((u, v, z))

        # Nahe Punkte zusammenfassen mit DBSCAN
        merged_features = self.merge_close_points(valid_features, eps=8, min_samples=1)

        # Ausgabe-Bild
        img_out = frame.copy()

        # PointCloud-Punkte
        points_xyz = []

        for (u, v, z) in merged_features:
            u_i, v_i = int(round(u)), int(round(v))

            # Winkel berechnen
            angle_x = math.atan((u - self.cx) / self.fx)
            angle_y = math.atan((v - self.cy) / self.fy)

            # 3D berechnen
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            points_xyz.append([float(x), float(y), float(z)])

            # Punkt einzeichnen
            cv2.circle(img_out, (u_i, v_i), 2, (0, 255, 0), -1)

            # Tiefe daneben schreiben
            cv2.putText(
                img_out,f"{z:.2f}", (u_i + 3, v_i),cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 0),1)

        cv2.imshow("Depth_interest Points", img_out)
        cv2.waitKey(1)

        # PointCloud publizieren
        self.publish_pointcloud(points_xyz, msg.header.frame_id)

        self.get_logger().info(
            f"Matches gesamt: {len(matches)}, "
            f"gültig vor DBSCAN: {len(valid_features)}, "
            f"nach DBSCAN: {len(merged_features)}"
        )

        # aktuelles Bild als vorheriges speichern
        self.prev_gray = gray

    def is_valid_depth(self, d):
        if d is None:
            return False
        if np.isnan(d):
            return False
        if np.isinf(d):
            return False
        if d <= 0:
            return False
        return True

    def is_depth_locally_stable(self, u, v, patch_size=3, max_std=0.03):
        half = patch_size // 2

        if (u - half < 0 or v - half < 0 or
                u + half >= self.prev_depth_img.shape[1] or
                v + half >= self.prev_depth_img.shape[0]):
            return False

        patch = self.prev_depth_img[v-half:v+half+1, u-half:u+half+1].astype(np.float32)

        if self.depth_in_mm:
            patch /= 1000.0

        valid = np.isfinite(patch) & (patch > 0.0)
        vals = patch[valid]

        if len(vals) < 5:
            return False

        return np.std(vals) < max_std

    def merge_close_points(self, valid_features, eps=8, min_samples=1):
        """
        Fasst nahe Punkte im Bild zusammen.
        Geclustert wird in 2D über (u, v).
        Pro Cluster wird der Mittelwert von u, v, z genommen.
        """
        if not valid_features:
            return []

        uv = np.array([[p[0], p[1]] for p in valid_features], dtype=np.float32)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(uv)

        merged = []
        unique_labels = set(labels)

        for label in unique_labels:
            cluster_points = [valid_features[i] for i in range(len(valid_features)) if labels[i] == label]

            u_mean = float(np.mean([p[0] for p in cluster_points]))
            v_mean = float(np.mean([p[1] for p in cluster_points]))
            z_mean = float(np.mean([p[2] for p in cluster_points]))

            merged.append((u_mean, v_mean, z_mean))

        return merged

    def publish_pointcloud(self, points_xyz, frame_id):
        if not points_xyz:
            self.get_logger().warn('Keine gültigen 3D-Punkte zum Publizieren.')
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        # frame link setzen
        header.frame_id = frame_id if frame_id else 'nav_rgbd_1_depth_optical_frame'

        cloud_msg = point_cloud2.create_cloud_xyz32(header, points_xyz)
        self.pc_pub.publish(cloud_msg)


def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()