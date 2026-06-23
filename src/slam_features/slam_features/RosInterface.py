import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from nav_msgs.msg import Odometry
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import numpy as np
import math
from .FastSlam import VisualSLAMCore


class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', 
                                                                                     rclpy.parameter.Parameter.Type.BOOL, True)])
        
        self.bridge = CvBridge()
        
        # Initialize Visual SLAM
        self.slam = VisualSLAMCore()

        # Subscribe to RGB image topic
        self.subscription_rgb = self.create_subscription(Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)

        # Subscribe to depth image topic
        self.subscription_depth = self.create_subscription(Image, '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)

        # Subscribe to Wheel_Odom for initial output rotation and translatation
        self.wheel_sub = self.create_subscription(Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)

        # Publisher for 3D pointcloud
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)

        # Odometry Publisher
        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)

        # TF initialization
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Number of Frames skipped after update
        self.frame_counter = self.slam.config.frame_counter
        self.frame_index = 0

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None
        self.depth_frame = None

        # Initial orientation from wheel odometry
        self.wheel_start_theta = 0.0
        self.wheel_start_x = 0.0
        self.wheel_start_y = 0.0



    def wheel_callback(self, msg):

        q = msg.pose.pose.orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

        # Initialization for output rotation & translatation
        if self.wheel_start_theta == 0.0:   self.wheel_start_theta = yaw
        if self.wheel_start_x == 0.0:       self.wheel_start_x = x
        if self.wheel_start_y == 0.0:       self.wheel_start_y = y


    
    def rotation_matrix_2d(self, theta):
        c = math.cos(theta)
        s = math.sin(theta)

        return np.array([[c, -s], [s, c]], dtype=np.float64)



    def lookup_static_tf(self):
        # Load static Kinect-to-base transform and initialize transformation matrices
        if self.kinect_to_base_matrix is None:
            try:

                # get TF from kinect_depth to base_link
                t = self.tf_buffer.lookup_transform(self.base_frame, 'kinect_depth', rclpy.time.Time())
                
                self.kinect_to_base_matrix = np.eye(4)
                
                # store translation
                self.kinect_to_base_matrix[0, 3] = t.transform.translation.x * 1000 # convert to mm
                self.kinect_to_base_matrix[1, 3] = t.transform.translation.y * 1000 # convert to mm
                self.kinect_to_base_matrix[2, 3] = t.transform.translation.z * 1000 # convert to mm

                # convert quaternion to rotation
                quat = [t.transform.rotation.x, t.transform.rotation.y, 
                        t.transform.rotation.z, t.transform.rotation.w]
                
                self.kinect_to_base_matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
                self.base_to_kinect_matrix = np.linalg.inv(self.kinect_to_base_matrix)
                return True
            
            except Exception as e:
                # TF is not available -> log error
                self.get_logger().info(f"wait for static TF ... {e}")
                return False
            
        return True



    def listener_callback_depth(self, msg):
        # getting deph of Image
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')



    def listener_callback_rgb(self, msg):
        # static TF not loaded
        if not self.lookup_static_tf() or self.depth_frame is None:     return
        
        if self.frame_counter <=0:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Slam processing in FastSLAM.py
            pose_updated, pose, best_map_manager = self.slam.process_frame(frame, self.depth_frame, self.kinect_to_base_matrix,
                                                                           self.base_to_kinect_matrix, self.frame_index)

            if pose_updated:
                self.publish_tf(pose.x / 1000.0, pose.y / 1000.0, pose.theta, msg.header.stamp)

                # publish only each 10th Pose update (improved Performance)
                if self.frame_index % 10 == 0:
                    self.publish_robots_tf_array(self.slam.robots, msg.header.stamp)

                self.publish_odometry_msg(pose.x / 1000.0, pose.y / 1000.0, pose.theta, msg.header.stamp)
                self.frame_counter = self.slam.config.frame_counter

            # Publish Landmarks as PointCloud2
            header = std_msgs.msg.Header()
            header.stamp = msg.header.stamp
            header.frame_id = self.odom_frame
            map_points = best_map_manager.get_all_points_for_msg()

            # Check for empty Map
            if map_points:

                # compensate initiale Map rotation
                R = self.rotation_matrix_2d(-self.wheel_start_theta)
                rotated_points = []

                for p in map_points:
                    xy = np.array([p[0], p[1]])
                    xy_rot = xy @ R

                    # Add initial translatation Error from wheel Odom
                    rotated_points.append([xy_rot[0] + self.wheel_start_x, xy_rot[1]+ self.wheel_start_y, p[2]])

                self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, rotated_points))

        self.frame_counter -= 1
        self.frame_index   += 1



    def publish_tf(self, x, y, theta, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        # Vektor for Odometrie rotation
        v = [x, y]
        v = v @ self.rotation_matrix_2d(-self.wheel_start_theta)

        # correction of translation error from wheel Odom
        t.transform.translation.x = v[0] + self.wheel_start_x
        t.transform.translation.y = v[1] + self.wheel_start_y
        t.transform.translation.z = 0.0

        euler = Rotation.from_euler('z', float(theta) + self.wheel_start_theta)
        quat = euler.as_quat(canonical=True)

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    # Publish all virtual particle poses as TF frames for visualization.
    def publish_robots_tf_array(self, robots_list, stamp):
        
        tf_messages = []

        for robot in robots_list:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = f"virtual_robot_{robot.id}"

            # Vector for Odometrie rotation
            v = [robot.pose.x, robot.pose.y]
            v = v @ self.rotation_matrix_2d(-self.wheel_start_theta)

            # correction of translation error from wheel Odom
            t.transform.translation.x = (v[0] + self.wheel_start_x) / 1000.0
            t.transform.translation.y = (v[1] + self.wheel_start_y) / 1000.0
            t.transform.translation.z = 0.0

            theta = robot.pose.theta
            euler = Rotation.from_euler('z', float(theta) + self.wheel_start_theta)
            quat = euler.as_quat(canonical=True)

            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            tf_messages.append(t)

        if tf_messages:     self.tf_broadcaster.sendTransform(tf_messages)



    def publish_odometry_msg(self, x, y, theta, stamp):
        msg = Odometry()

        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame

        # Vektor for Odometrie rotation
        v = [x, y]
        v = v @ self.rotation_matrix_2d(-self.wheel_start_theta)

        # correction of translation error from wheel Odom
        msg.pose.pose.position.x = v[0] + self.wheel_start_x
        msg.pose.pose.position.y = v[1] + self.wheel_start_y
        msg.pose.pose.position.z = 0.0

        r = Rotation.from_euler('z', float(theta) + self.wheel_start_theta)
        quat = r.as_quat(canonical=True)

        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        self.odom_publisher.publish(msg)

    def destroy_node(self):
        self.slam.shutdown()
        super().destroy_node()