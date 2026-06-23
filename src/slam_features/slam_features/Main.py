#!/usr/bin/env python3
import rclpy
import cProfile
from .RosInterface import SlamNode

"""
ROS 2 Entry Point

Algorithm Overview:
  - Particle filter over robot poses
  - ORB feature extraction and matching
  - Kabsch algorithm + RANSAC 
  - EKF per landmark 

File Structure:
  Main.py            - Entry point; starts the ROS 2 node

  RosInterface.py    - ROS 2 node (SlamNode): subscribes to RGB/Depth topics,
                       publishes TF, odometry, and PointCloud2

  FastSlam.py        - Core logic: particle filter, resampling, keyframe
                       management, loop closure detection (VisualSLAMCore)
                       and particle class

  EKF.py             - Extended Kalman Filter per landmark,
                       Landmark dataclass, and map management 

  geometry_utils.py  - Geometry helper functions: Kabsch-2D, RANSAC,
                       2D to 3D projection, landmark visibility test

  Configuration.py   - All parameters: camera intrinsics, RANSAC, EKF noise,
                       particle filter thresholds
"""


def main(args=None):
    profiler = cProfile.Profile()
    profiler.enable()

    rclpy.init(args=args)
    node = SlamNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("Ctrl+C erkannt")

    finally:
        profiler.disable()
        profiler.dump_stats("/home/mathias/profiling_result.prof")
        print("Profil gespeichert")

        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()