from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robile_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
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
            "exploration_node = robile_explorer.exploration_node:main",           
        ],
    },
)
