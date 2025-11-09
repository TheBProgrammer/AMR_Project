from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Optional: keep these if you use them elsewhere
    pkg = get_package_share_directory('robile_path_planner')
    description_pkg = get_package_share_directory('robile_description')

    use_sim_time = LaunchConfiguration('use_sim_time')
    localization_source = LaunchConfiguration('localization_source')

    return LaunchDescription([
        # Use sim time (Gazebo) or wall time (real robot)
        DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true',
            description='Use simulation time if true'
        ),

        # expose localization source as a launch arg
        # Valid choices: amcl | mcl | slam | odom
        DeclareLaunchArgument(
            name='localization_source',
            default_value='amcl',
            description='Localization source to use: amcl | mcl | slam | odom'
        ),

        Node(
            package='robile_path_planner',
            executable='navigation_manager',
            name='navigation_manager',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'localization_source': localization_source
            }]
        ),
    ])