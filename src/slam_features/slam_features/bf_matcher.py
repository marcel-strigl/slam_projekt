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
        self.subscription = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.orb_callback, 10)
        self.img2 = None
        self.rgb2 = None
        
    def orb_callback(self,msg):
        frame = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.img2 is None:
            self.img2 = img1
            self.rgb2 = frame
            return

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(self.img2,None)
        
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(frame,kp1,self.rgb2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv2.putText(img3,500,cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.imshow('matches',img3)
        cv2.waitKey(1)
        self.img2 = img1
        self.rgb2 = frame

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()