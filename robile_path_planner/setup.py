from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'robile_path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
        
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thebrobot',
    maintainer_email='bhaveshgandhi843@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "a_star_planner = robile_path_planner.a_star:main",
            "extract_waypoints = robile_path_planner.extract_waypoints:main",
            "field_based_planner = robile_path_planner.field_based_planner:main",
            'navigation_manager = robile_path_planner.navigation_manager:main',
        ],
    },
)
