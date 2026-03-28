from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():

    # ORB Node
    orb_node = Node(
        package='slam_features',
        executable='orb',
        name='orb',
        output='screen',
    )

    # Bag abspielen
    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            '/home/marcel/SLAM_Projekt/20260324_Project_Bags/pure_rotation_bag/pure_rotation_bag',
            '--clock'
        ],
        output='screen'
    )

    return LaunchDescription([
        orb_node,
        bag_play
    ])