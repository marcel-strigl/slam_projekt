#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sklearn

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('orb')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/serf01/nav_rgbd_1/rgb/image_raw',
            self.bf_callback,
            10
        )

        self.subscription_depth = self.create_subscription(
            Image,
            '/serf01/nav_rgbd_1/depth/image_raw',
            self.depth_callback,
            10
        )

        self.depth_img = None

    def bf_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(gray, None)

        img_out = frame.copy()

        if self.depth_img is not None:
            h, w = self.depth_img.shape

            for point in kp:
                x, y = point.pt
                x, y = int(x), int(y)

                if 0 <= x < w and 0 <= y < h:
                    depth = self.depth_img[y, x]
                    
                    if depth > 0 and depth <= 20000: 
                        depth_m = depth / 1000.0
                        # Punkt zeichnen
                        cv2.circle(img_out, (x, y), 2, (0, 255, 0), -1)
                        # Text 
                        cv2.putText(img_out,f"{depth_m:.2f}",(x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1)

        cv2.imshow("Depth_interest Points", img_out)
        cv2.waitKey(1)


    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        

def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()