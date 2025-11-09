# particle_filter.launch.py

from launch import LaunchDescription
from launch.actions import TimerAction, RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart

def generate_launch_description():
    # Map server node (lifecycle managed)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': '/home/thebrobot/amr_prj_ws/src/robile_localization/maps/olab.yaml',
            'use_sim_time': False
        }]
    )

    # Lifecycle manager node to activate map_server
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )

    # Particle filter node with parameter to wait for initial pose from user before running
    # particle_filter_node = Node(
    #     package='robile_localization',
    #     executable='particle_filter',
    #     name='particle_filter',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #         'wait_for_initial_pose': True  # Your PF node should handle this param accordingly
    #     }]
    # )

    particle_filter_node = Node(
        package='robile_localization',
        executable='mcl_node',
        name='mcl_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'base_frame': "base_link",
            'wait_for_initial_pose': True  # Your PF node should handle this param accordingly
        }]
    )

    # Delay launching particle_filter until after map_server lifecycle_manager starts
    # Using event handler: start particle_filter once lifecycle_manager_map is up
    start_pf_after_lifecycle = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=lifecycle_manager_node,
            on_start=[particle_filter_node]
        )
    )

    return LaunchDescription([
        map_server_node,
        lifecycle_manager_node,
        start_pf_after_lifecycle,
    ])
