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

## Changes to the front-end processing files for Florence 2

- Navigate to the processing_florence2.py file. You should be able to find it in the huggingface home folder. If using the ```base``` model instead of the ```large``` model change path appropriately from the ```microsoft``` directory
```
cd /modules/transformers_modules/microsoft/Florence-2-large/21a599d414c4d928c9032694c424fb94458e3594/processing_florence2.py
```
- Go to line 104 that says ```<CAPTION_TO_PHRASE_GROUNDING>``` and change it to following:
```
'<CAPTION_TO_PHRASE_GROUNDING>': "description_with_bboxes"
```
- Go to line 126 that gives the descriptive tasks prompt for the ```<CAPTION_TO_PHRASE_GROUNDING>``` task and change it to the following:
```
'<CAPTION_TO_PHRASE_GROUNDING>': "Locate the objects in the caption: {input}, in the image."
```
