# Vision utilities for ROS 2


Implements:
- object detection and tracking
- VLM queries


## Install 


```sh
cd ~/ws
git clone https://github.com/ZacRavichandran/vision-ros2 src
wget -O ./vision-ros2/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
colcon build --symlink-install 
```

## Running 

On the jackal 

```
ros2 launch vision_ros2 grounding_dino.launch.py
```