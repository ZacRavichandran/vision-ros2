from glob import glob

from setuptools import find_packages, setup

package_name = "vision_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/weights", glob("weights/*")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/resource", glob("resource/*.py")),
    ],
    install_requires=["setuptools", "rclpy"],
    zip_safe=True,
    maintainer="zacravi",
    maintainer_email="zachary.ravichandran@gmail.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grounding_dino_node = vision_ros2.grounding_dino_node:main",
        ],
    },
)
