#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('orb')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,
'/serf01/nav_rgbd_1/rgb/image_raw', self.orb_callback, 10)
        
    def orb_callback(self,msg):
        frame = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp = orb.detect(img,None)
        img2 =cv2.drawKeypoints(frame, kp, None, color=(255,0,0),flags=0)
        cv2.imshow("orb",img2)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()