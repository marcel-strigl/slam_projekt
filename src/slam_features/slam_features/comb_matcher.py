#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('matcher_inkl_angels')
        self.bridge = CvBridge()

        self.prev_gray = None
        self.depth_img = None
        
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw',self.rgb_callback,10)
        
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw',self.depth_callback,10)

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Erstes Bild nur speichern
        if self.prev_gray is None:
            self.prev_gray = gray
            return

        if self.depth_img is None:
            self.prev_gray = gray
            return

        # ORB Features
        kp1, des1 = self.orb.detectAndCompute(gray, None)
        kp2, des2 = self.orb.detectAndCompute(self.prev_gray, None)

        # Matching
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        h, w = self.depth_img.shape[:2]

        # gültige Punkte vor dem Zusammenfassen: (u, v, z)
        valid_features = []

        for m in matches:
            kp = kp1[m.queryIdx]   # Punkt im aktuellen Bild
            u, v = kp.pt
            u_i, v_i = int(round(u)), int(round(v))

            if not (0 <= u_i < w and 0 <= v_i < h):
                continue

            depth_raw = self.depth_img[v_i, u_i]

            if not self.is_valid_depth(depth_raw):
                continue

            z = depth_raw / 1000.0

            # sinnvoller Tiefenbereich
            if z < 0.2 or z > 20.0:
                continue

            valid_features.append((u, v, z))

        # Nahe Punkte zusammenfassen mit DBSCAN
        merged_features = self.merge_close_points(valid_features, eps=8, min_samples=1)

        # Ausgabe-Bild
        img_out = frame.copy()

        for (u, v, z) in merged_features:
            u_i, v_i = int(round(u)), int(round(v))

            # Punkt einzeichnen
            cv2.circle(img_out, (u_i, v_i), 2, (0, 255, 0), -1)

            # Tiefe daneben schreiben
            cv2.putText(img_out,f"{z:.2f}",(u_i + 3, v_i),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1)

        cv2.imshow("Depth_interest Points", img_out)
        cv2.waitKey(1)

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
                u + half >= self.depth_img.shape[1] or
                v + half >= self.depth_img.shape[0]):
            return False

        patch = self.depth_img[v-half:v+half+1, u-half:u+half+1].astype(np.float32)

        patch = patch/1000.0

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


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()