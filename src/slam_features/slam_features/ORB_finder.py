#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('feature_memory')
        self.bridge = CvBridge()

        # Kameraparameter
        self.cx = 318.525
        self.cy = 241.181
        self.fx = 526.61
        self.fy = 526.61

        self.depth_img = None
        self.frame_count = 0

        # Referenzbild 
        self.reference_frozen = False
        self.ref_gray = None
        self.ref_rgb = None
        self.ref_kp = None
        self.ref_des = None
        self.ref_points_3d = None

        self.freeze_frame = 100

        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw',self.rgb_callback,10)

        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw',self.depth_callback,10)

    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1

        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None:
            return

        # Referenzbild einmal einfrieren
        if not self.reference_frozen and self.frame_count >= self.freeze_frame:
            self.freeze_reference(gray, frame, kp, des)
            return

        # Aktuelle Features gegen Referenz matchen
        if self.ref_des is None or len(self.ref_des) == 0:
            return
        matches = self.bf.match(des, self.ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        img_out = frame.copy()
        good_matches = 0

        for m in matches:
            kp_curr = kp[m.queryIdx]
            u, v = kp_curr.pt
            u_i, v_i = int(round(u)), int(round(v))

            if not self.in_image(u_i, v_i, self.depth_img):
                continue

            depth_raw = self.depth_img[v_i, u_i]

            if not self.is_valid_depth(depth_raw):
                continue

            z = float(depth_raw) / 1000.0

            # Plausibilitätsbereich laut Aufgabe: nicht 0, nicht größer als 4m
            if z <= 0.0 or z > 4.0:
                continue

            good_matches += 1

            cv2.circle(img_out, (u_i, v_i), 3, (0, 255, 0), -1)
            cv2.putText(img_out,f"{z:.2f}m",(u_i + 4, v_i),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0, 255, 0),1)

        cv2.putText(
            img_out,
            f"matches to frozen reference: {good_matches}",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

        cv2.imshow("Current image with recovered reference features", img_out)
        cv2.imshow("Frozen reference image", self.ref_rgb)
        cv2.waitKey(1)

    def freeze_reference(self, gray, frame, kp, des):
        valid_kp = []
        valid_des = []
        valid_points_3d = []

        h, w = self.depth_img.shape[:2]

        for i, keypoint in enumerate(kp):
            u, v = keypoint.pt
            u_i, v_i = int(round(u)), int(round(v))

            if not (0 <= u_i < w and 0 <= v_i < h):
                continue

            depth_raw = self.depth_img[v_i, u_i]

            if not self.is_valid_depth(depth_raw):
                continue

            z = float(depth_raw) / 1000.0

            if z <= 0.0 or z > 4.0:
                continue

            # 3D-Rekonstruktion
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            valid_kp.append(keypoint)
            valid_des.append(des[i])
            valid_points_3d.append([float(x), float(y), float(z)])

        if len(valid_des) == 0:
            self.get_logger().warn('Keine plausibilisierten Referenzfeatures gefunden.')
            return

        self.ref_gray = gray.copy()
        self.ref_rgb = frame.copy()
        self.ref_kp = valid_kp
        self.ref_des = np.array(valid_des, dtype=np.uint8)
        self.ref_points_3d = np.array(valid_points_3d, dtype=np.float32)
        self.reference_frozen = True

        # Referenzbild markieren
        ref_vis = frame.copy()

        for idx, (kp_ref, p3d) in enumerate(zip(self.ref_kp, self.ref_points_3d)):
            u, v = kp_ref.pt
            u_i, v_i = int(round(u)), int(round(v))
            z = p3d[2]

            # ORB-Punkt sichtbar markieren
            cv2.circle(ref_vis, (u_i, v_i), 4, (0, 255, 0), 1)
            cv2.circle(ref_vis, (u_i, v_i), 1, (0, 255, 0), -1)

            # Tiefe daneben
            cv2.putText(ref_vis,f"{z:.2f}m",(u_i + 5, v_i - 3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0, 255, 0),1)

        self.ref_rgb = ref_vis
    
        cv2.imshow("Frozen reference image", self.ref_rgb)
        cv2.waitKey(1)

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

    def in_image(self, u, v, image):
        h, w = image.shape[:2]
        return 0 <= u < w and 0 <= v < h

def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()