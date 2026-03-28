#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,
'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)
    def listener_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        cv2.imshow("Image",frame)
        cv2.waitKey(1)
def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()