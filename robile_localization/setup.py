from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robile_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        # Install map files
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        
        # Install rviz files
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thebrobot',
    maintainer_email='bhaveshgandhi843@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "particle_filter= robile_localization.particle_filter:main",
            "mcl_node = robile_localization.mcl_node:main"
        ],
    },
)
