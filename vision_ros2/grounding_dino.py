#!/usr/bin/env python3

import time
from typing import List, Optional, Tuple

import numpy as np
import sys
import warnings

import numpy as np
import torch
import torchvision


try:
    from groundingdino.util.inference import Model as GDModel

    from vision_ros2.utils import vis_result_fast
except ImportError as ex:
    import sys
    raise ValueError(f"must install grounding dino: {ex}\n"
                     f"current sys: {sys.executable}, {sys.path}")


warnings.filterwarnings("ignore", message="Unable to import Axes3D")


class GroundingDinoInfer:
    def __init__(
        self,
        classes: List[str],
        confidence: float,
        ckpt_path: str,
        config_path: str,
        device: Optional[str] = "cuda",
    ):
        """Grounding Dino Inference.

        Parameters
        ----------
        classes : List[str]
            Classes to predict
        confidence : float
            Confidence threshold for predictions
        ckpt_path : str
            Path to model checkpoint.
        config_path : str
            Path to model config
        device : Optional[str] = cuda
            Device on which to put model
        """
        self.grounding_dino_model = GDModel(
            model_config_path=str(config_path),
            model_checkpoint_path=str(ckpt_path),
            device=device,
        )
        self.classes = classes
        self.confidence = confidence
        self.grounding_dino_model.model.eval()

    def set_labels(self, labels: List[str]) -> bool:
        self.classes = labels
        return True

    def predict(
        self, img: np.ndarray, plot_output: Optional[bool] = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on `img` using Grounding Dino.

        Parameters
        ----------
        img : np.ndarray
            Incoming image in (h, w, c)
        plot_output : Optional[bool], optional
            True to return annotated image w/ detections.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - annotated image (input if `plot_output` is false)
            - classes
            - boxes in xyxy
            - confidences
        """
        pred_classes = self.classes.copy()

        detections = self.grounding_dino_model.predict_with_classes(
            img,
            pred_classes,
            box_threshold=self.confidence,
            text_threshold=self.confidence,
        )

        annotated_image = img

        if len(detections.class_id) > 0:
            ### Non-maximum suppression ###
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    0.5,
                )
                .numpy()
                .tolist()
            )

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Somehow some detections will have class_id=-1, remove them
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

            if plot_output:
                annotated_image, labels = vis_result_fast(
                    img, detections, pred_classes, instance_random_color=True
                )

            label_str = [pred_classes[cid] for cid in detections.class_id]

            return (
                annotated_image,
                label_str,
                detections.xyxy,
                detections.confidence,
            )
        else:
            return img, "", np.array([]), np.array([])