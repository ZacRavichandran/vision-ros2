#!/usr/bin/env python3

import argparse
import os
import sys
import warnings
from typing import List, Optional, Tuple

import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image, ImageDraw, ImageFont

try:
    from groundingdino.util.inference import Model as GDModel

    from vision_ros2.utils import vis_result_fast
except ImportError as ex:
    import sys

    raise ValueError(
        f"must install grounding dino: {ex}\n"
        f"current sys: {sys.executable}, {sys.path}"
    )


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
        # self.grounding_dino_model = GDModel(
        #     model_config_path=str(config_path),
        #     model_checkpoint_path=str(ckpt_path),
        #     device=device,
        # )
        self.grounding_dino_model = self.load_model(
            model_config_path=str(config_path), model_checkpoint_path=str(ckpt_path)
        )
        self.classes = classes
        self.confidence = confidence
        # self.grounding_dino_model.model.eval()

    def load_model(self, model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def preprocess_img(self, img: np.ndarray):
        image_pil = Image.fromarray(img).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def set_labels(self, labels: List[str]) -> bool:
        self.classes = labels
        return True

    def get_grounding_output(
        self,
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        with_logits=True,
        cpu_only=False,
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        pred_logits = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            pred_phrases.append(pred_phrase)
            pred_logits.append(logit.max().item())
            # if with_logits:
            #     pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            # else:
            #     pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases, pred_logits

    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        return image_pil, mask

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

        img_pil, img = self.preprocess_img(img)

        boxes, labels, confidences = self.get_grounding_output(
            self.grounding_dino_model,
            image=img,
            caption=",".join(pred_classes),
            box_threshold=self.confidence,
            text_threshold=self.confidence,
        )

        if len(boxes) == 0:
            return img, "", np.array([]), np.array([])

        size = img_pil.size
        pred_dict = {
            "boxes": boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": labels,
        }
        annotated_img = self.plot_boxes_to_image(img_pil, pred_dict)[0]

        # detections = self.grounding_dino_model.predict_with_classes(
        #     img,
        #     pred_classes,
        #     box_threshold=self.confidence,
        #     text_threshold=self.confidence,
        # )

        # annotated_image = img

        # if len(detections.class_id) > 0:
        #     ### Non-maximum suppression ###
        #     nms_idx = (
        #         torchvision.ops.nms(
        #             torch.from_numpy(detections.xyxy),
        #             torch.from_numpy(detections.confidence),
        #             0.5,
        #         )
        #         .numpy()
        #         .tolist()
        #     )

        #     detections.xyxy = detections.xyxy[nms_idx]
        #     detections.confidence = detections.confidence[nms_idx]
        #     detections.class_id = detections.class_id[nms_idx]

        #     # Somehow some detections will have class_id=-1, remove them
        #     valid_idx = detections.class_id != -1 and detections.class_id != None
        #     detections.xyxy = detections.xyxy[valid_idx]
        #     detections.confidence = detections.confidence[valid_idx]
        #     detections.class_id = detections.class_id[valid_idx]

        #     if plot_output:
        #         annotated_image, labels = vis_result_fast(
        #             img, detections, pred_classes, instance_random_color=True
        #         )

        #     label_str = [pred_classes[cid] for cid in detections.class_id]

        return (np.asarray(annotated_img), labels, boxes, confidences)
