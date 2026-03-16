#!/usr/bin/env python3

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")


class SAM3Infer:
    def __init__(
        self,
        classes: List[str],
        confidence: float,
        ckpt_path: str,
        device: Optional[str] = "cuda",
    ):
        """SAM3 (Segment Anything Model 3) Inference.

        Analogous to GroundingDinoInfer, but returns segmentation masks in
        addition to bounding boxes. Uses SAM3SemanticPredictor under the hood
        for open-vocabulary, text-prompted segmentation.

        NOTE: SAM3 weights (sam3.pt) require manual download after requesting
        access at https://huggingface.co/models?search=sam3

        Parameters
        ----------
        classes : List[str]
            Classes to predict (text prompts, e.g. ["person", "red car"])
        confidence : float
            Confidence threshold for predictions
        ckpt_path : str
            Path to sam3.pt model weights
        device : Optional[str]
            Device on which to run the model ("cuda", "cpu", "mps")
        """
        self.classes = classes
        self.confidence = confidence
        self.device = device
        self.predictor = self.load_model(ckpt_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, ckpt_path: str):
        """Load and return a SAM3SemanticPredictor instance."""
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=self.confidence,
            task="segment",
            mode="predict",
            model=ckpt_path,
            device=self.device,
            half=(self.device == "cuda"),  # FP16 on GPU for speed
            verbose=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print(f"[SAM3Infer] Loaded SAM3 from '{ckpt_path}' on device '{self.device}'")
        return predictor

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess_img(
        self, img: np.ndarray, camera_name: str
    ) -> Tuple[Image.Image, np.ndarray]:
        """Convert a raw numpy image to a PIL image, applying camera-specific
        transforms (mirrors GroundingDinoInfer.preprocess_img).

        Parameters
        ----------
        img : np.ndarray
            Input image in HWC uint8 format
        camera_name : str
            Camera identifier; "spot_camera" triggers a 90° CCW rotation

        Returns
        -------
        Tuple[Image.Image, np.ndarray]
            - PIL image (for annotation / size queries)
            - numpy array (passed directly to SAM3 — no manual normalisation
              needed; the predictor handles that internally)
        """
        image_pil = Image.fromarray(img).convert("RGB")
        image_np = np.asarray(image_pil)
        return image_pil, image_np

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def set_labels(self, labels: List[str]) -> bool:
        """Update the list of text-prompt classes."""
        self.classes = labels
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_sam3_output(
        self,
        image_np: np.ndarray,
        classes: List[str],
    ) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[float]]:
        """Run SAM3 semantic inference and collect per-detection results.

        SAM3SemanticPredictor returns one Results object per text prompt
        (i.e. one per class). Each Results object may contain multiple
        instances of that class.

        Parameters
        ----------
        image_np : np.ndarray
            RGB image as a numpy array
        classes : List[str]
            Text prompts to query

        Returns
        -------
        Tuple[List[np.ndarray], List[str], List[np.ndarray], List[float]]
            - boxes_list   : list of [N, 4] xyxy arrays (one per detection)
            - labels_list  : list of class-name strings
            - masks_list   : list of (H, W) boolean mask arrays
            - confs_list   : list of confidence floats
        """
        self.predictor.set_image(image_np)
        results_per_class = self.predictor(text=classes)

        boxes_list: List[np.ndarray] = []
        labels_list: List[str] = []
        masks_list: List[np.ndarray] = []
        confs_list: List[float] = []

        for class_name, result in zip(classes, results_per_class):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xywh = result.boxes.xywh.cpu().numpy()  # (N, 4)
            confs = result.boxes.conf.cpu().numpy()  # (N,)
            labels = result.boxes.cls.cpu().numpy().astype(np.uint8)

            masks = None
            if result.masks is not None:
                # masks.data: (N, H, W) float32 logits → threshold to bool
                masks = result.masks.data.cpu().numpy() > 0.5  # (N, H, W)

            for i in range(len(boxes_xywh)):
                boxes_list.append(boxes_xywh[i])
                labels_list.append(classes[labels[i]])
                confs_list.append(float(confs[i]))
                masks_list.append(masks[i] if masks is not None else None)

        return boxes_list, labels_list, masks_list, confs_list

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_detections_to_image(
        self, image_pil: Image.Image, tgt: Dict[str, Any]
    ) -> Tuple[Image.Image, Image.Image]:
        """Draw masks and bounding boxes onto `image_pil`.

        Analogous to GroundingDinoInfer.plot_boxes_to_image, but also
        renders semi-transparent segmentation masks when available.

        Parameters
        ----------
        image_pil : Image.Image
            Source PIL image
        tgt : Dict[str, Any]
            Must contain:
                "boxes"       – list of [4] xywh numpy arrays
                "labels"      – list of strings
                "confidences" – list of floats
                "masks"       – list of (H, W) bool arrays or None entries

        Returns
        -------
        Tuple[Image.Image, Image.Image]
            - annotated RGB image
            - binary detection mask (white = any detection)
        """
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        confs = tgt["confidences"]
        masks = tgt["masks"]
        assert len(boxes) == len(labels), "boxes and labels must have the same length"

        # Work on a copy so the original is not mutated
        annotated = image_pil.copy().convert("RGBA")
        mask_img = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        draw = ImageDraw.Draw(annotated)

        for box, label, conf, seg_mask in zip(boxes, labels, confs, masks):
            color = tuple(np.random.randint(60, 220, size=3).tolist())

            # --- segmentation mask overlay --------------------------------
            if seg_mask is not None:
                # seg_mask: (H, W) bool
                mask_rgba = np.zeros(
                    (seg_mask.shape[0], seg_mask.shape[1], 4), dtype=np.uint8
                )
                mask_rgba[seg_mask] = (*color, 100)  # semi-transparent fill
                mask_overlay = Image.fromarray(mask_rgba, mode="RGBA")
                # Resize to match image in case of scale difference
                if mask_overlay.size != image_pil.size:
                    mask_overlay = mask_overlay.resize(image_pil.size, Image.NEAREST)
                annotated = Image.alpha_composite(annotated, mask_overlay)
                draw = ImageDraw.Draw(annotated)  # refresh after composite

            # --- bounding box ---------------------------------------------
            x, y, w, h = box
            # x0, y0, x1, y1 = (int(v) for v in box)

            x0 = int(x - w / 2)
            y0 = int(y - h / 2)
            x1 = int(x + w / 2)
            y1 = int(y + h / 2)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=3)

            # --- label text -----------------------------------------------
            plot_label = f"{label}_{conf:.2f}"
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                text_bbox = draw.textbbox((x0, y0), plot_label, font)
            else:
                w, h = draw.textsize(plot_label, font)
                text_bbox = (x0, y0, x0 + w, y0 + h)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x0, y0), plot_label, fill="white", font=font)

        # Convert back to RGB for downstream compatibility
        annotated_rgb = annotated.convert("RGB")
        return annotated_rgb, mask_img

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(
        self,
        img: np.ndarray,
        plot_output: Optional[bool] = True,
        camera_name: Optional[str] = "zed_camera",
    ) -> Tuple[np.ndarray, List[str], List[np.ndarray], List[float], List[np.ndarray]]:
        """Run SAM3 inference on `img`.

        Parameters
        ----------
        img : np.ndarray
            Input image in (H, W, C) uint8 format
        plot_output : Optional[bool]
            True to return an annotated image with detections drawn
        camera_name : Optional[str]
            Camera identifier forwarded to preprocess_img

        Returns
        -------
        Tuple[np.ndarray, List[str], List[np.ndarray], List[float], List[np.ndarray]]
            - annotated_img : (H, W, 3) uint8 numpy array
                             (unannotated input image if plot_output is False
                             or no detections found)
            - labels        : list of detected class-name strings
                             (empty string "" if nothing detected)
            - boxes         : list of [4] xyxy numpy arrays
                             (empty array if nothing detected)
            - confidences   : list of confidence floats
                             (empty array if nothing detected)
            - masks         : list of (H, W) bool numpy arrays
                             (empty array if nothing detected)
        """
        pred_classes = self.classes.copy()
        img_pil, img_np = self.preprocess_img(img, camera_name=camera_name)

        boxes, labels, masks, confidences = self.get_sam3_output(img_np, pred_classes)

        if len(boxes) == 0:
            return np.asarray(img_pil), "", np.array([]), np.array([]), np.array([])

        if plot_output:
            tgt = {
                "boxes": boxes,
                "labels": labels,
                "confidences": confidences,
                "masks": masks,
            }
            annotated_pil, _ = self.plot_detections_to_image(img_pil, tgt)
            annotated_img = np.asarray(annotated_pil)
        else:
            annotated_img = np.asarray(img_pil)

        return annotated_img, labels, boxes, confidences
