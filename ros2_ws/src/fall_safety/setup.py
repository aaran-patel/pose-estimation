from setuptools import find_packages, setup

package_name = "fall_safety"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aaran Patel",
    maintainer_email="patelaarav2006@gmail.com",
    description="ROS2 node that detects falls from pose estimation and publishes a clear-to-move safety signal.",
    entry_points={
        "console_scripts": [
            "fall_safety_node = fall_safety.live_fall_alert:main",
        ],
    },
)
