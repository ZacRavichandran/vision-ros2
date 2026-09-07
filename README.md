# vision_ros2

ROS 2 (ament_python) package for onboard perception: object detection & tracking, camera stitching, and vision-language model (VLM) scene queries. Used on Clearpath Jackal/Husky and Boston Dynamics Spot platforms as part of the [SPINE](https://zacravichandran.github.io/SPINE) and [SPINE-HT](https://zacravichandran.github.io/SPINE-HT/) stacks

## Features

- **Detection & tracking** (`detector_node`) — runs [SAM3](https://github.com/facebookresearch/sam2) (SAM 3) or [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) open-vocabulary detection on one or more camera streams, projects detections to 3D using depth + odometry, and tracks them over time. Publishes `teaming_msgs/Detection` and `teaming_msgs/Track` messages plus RViz markers.
- **VLM scene queries** (`vlm_node`) — answers natural-language questions about the current camera view (e.g. scene classification, "why" explanations) using a VLM such as [VipLlava](https://huggingface.co/llava-hf/vip-llava-7b-hf) or Llava-Phi-3-mini, served via a `teaming_msgs/Query` service.
- **Camera stitching** (`stitcher_node`) — combines multiple camera feeds into a single panoramic image.

## Dependencies

- ROS 2 (tested with a recent distro; built with `ament_python` / `colcon`)
- [`teaming_msgs`](.) — custom message/service definitions (`Detection`, `Track`, `Query`, `GetLabels`, `SetLabels`, etc.). This is **not included in this repo** and must be built alongside it.
- Python packages: `torch`, `transformers`, `ultralytics` (SAM3), `groundingdino-py` (only if using the GroundingDINO backend), `opencv-python` / `cv_bridge`, `scipy`, `tf2_ros`
- SAM3 requires a specific CLIP version to load its text encoder correctly — follow the "Install" section of the [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3) rather than installing `ultralytics` + `clip` independently.

## Install

```sh
cd ~/ws/src
git clone https://github.com/ZacRavichandran/vision-ros2 vision_ros2
```

Download model weights for the detector backend you plan to use, e.g. GroundingDINO:

```sh
wget -O vision_ros2/weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

For SAM3, place your checkpoint anywhere on disk and point the `weights` parameter (see `config/*.yaml`) at it.

Then build:

```sh
cd ~/ws
colcon build --symlink-install
```

## Running

Launch files live in [launch/](launch/) and are configured via YAML files in [config/](config/) (`jackal.yaml`, `spot.yaml`, `spot_frontleft.yaml`, `spot_frontright.yaml`, `spot_right.yaml`, `spot_all_cameras.yaml`).

```sh
# Generic entrypoint, takes a config_file argument
ros2 launch vision_ros2 vision.launch.py config_file:=<path_to_config>.yaml

# Platform-specific launch files
ros2 launch vision_ros2 vision_clearpath.launch.py
ros2 launch vision_ros2 vision_spot.launch.py
ros2 launch vision_ros2 vision_sim.launch.py

# Static TF publishers for each platform
ros2 launch vision_ros2 vision_static_transforms.launch.py
ros2 launch vision_ros2 spot_static_transforms.launch.py
```

## License

MIT — see [LICENSE](LICENSE).
