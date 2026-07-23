import os
from glob import glob

from setuptools import find_packages, setup

package_name = "aura_pose_estimation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aaran Patel",
    maintainer_email="patelaarav2006@gmail.com",
    description="Pose estimation and fall detection nodes for the Aura project.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pose_estimation_node = aura_pose_estimation.pose_estimation_node:main",
            "fall_detection_node = aura_pose_estimation.fall_detection_node:main",
            "fall_visualizer_node = aura_pose_estimation.fall_visualizer_node:main",
        ],
    },
)
