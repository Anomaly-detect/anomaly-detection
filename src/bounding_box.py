"""
Bounding Box and Anomaly Spatial Localization Module
----------------------------------------------------
Location: d:/anomaly-detection/src/bounding_box.py

Provides utilities and an end-to-end extractor for generating spatial
bounding boxes around anomalous events from ASTNet reconstruction error maps.
"""

import os
import sys
from typing import List, Tuple, Dict, Optional, Union
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_spatial_error_map(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    method: str = "mse",
) -> np.ndarray:
    """
    Compute 2D spatial error map between predicted frame and target frame.

    Args:
        pred: Predicted frame tensor [B, 3, H, W] or [3, H, W] or numpy array.
              Values normalized in range [-1.0, 1.0] or [0.0, 1.0].
        target: Target frame tensor with matching shape to pred.
        method: 'mse' (squared difference) or 'l1' (absolute difference).

    Returns:
        error_map: 2D numpy array of shape [H, W] (or [B, H, W] if batch input).
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        # Convert range from [-1.0, 1.0] to [0.0, 1.0] if negative values are present
        if pred.min() < 0.0 or target.min() < 0.0:
            p_01 = (pred + 1.0) * 0.5
            t_01 = (target + 1.0) * 0.5
        else:
            p_01 = pred
            t_01 = target

        # Handle 4D [B, 3, H, W] vs 3D [3, H, W]
        dim = 1 if p_01.dim() == 4 else 0
        if method.lower() == "l1":
            diff = torch.mean(torch.abs(p_01 - t_01), dim=dim)
        else:
            diff = torch.mean((p_01 - t_01) ** 2, dim=dim)

        res = diff.detach().cpu().numpy()
        if res.ndim == 3 and res.shape[0] == 1:
            res = res.squeeze(0)
        return res

    # NumPy arrays
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)

    # Normalize if in [-1.0, 1.0]
    if p.min() < 0.0 or t.min() < 0.0:
        p = (p + 1.0) * 0.5
        t = (t + 1.0) * 0.5

    # Check channels axis: (C, H, W) vs (H, W, C)
    if p.ndim == 3:
        if p.shape[0] in (1, 3):  # (C, H, W)
            axis = 0
        else:  # (H, W, C)
            axis = 2
    elif p.ndim == 4:
        axis = 1  # (B, C, H, W)
    else:
        raise ValueError(f"Unsupported array dimensions: {p.ndim}")

    if method.lower() == "l1":
        res = np.mean(np.abs(p - t), axis=axis)
    else:
        res = np.mean((p - t) ** 2, axis=axis)

    if res.ndim == 3 and res.shape[0] == 1:
        res = res.squeeze(0)
    return res


def error_map_to_mask(
    error_map: np.ndarray,
    method: str = "otsu",
    threshold_ratio: float = 0.5,
    percentile: float = 98.0,
    blur_kernel: int = 7,
    morph_kernel_size: int = 5,
    dilation_iters: int = 2,
) -> np.ndarray:
    """
    Convert a 2D float error map into a cleaned binary foreground mask.

    Args:
        error_map: 2D float numpy array [H, W].
        method: 'otsu' (automatic bimodal threshold),
                'percentile' (top N% error pixels),
                or 'fixed' (threshold_ratio * max_val).
        threshold_ratio: Used when method is 'fixed' (range: 0.0 - 1.0).
        percentile: Percentile cutoff when method is 'percentile' (e.g. 98.0).
        blur_kernel: Kernel size for Gaussian blur smoothing (must be odd).
        morph_kernel_size: Kernel size for morphological closing & dilation.
        dilation_iters: Number of dilation iterations to bridge fragmented parts.

    Returns:
        binary_mask: uint8 numpy array [H, W] with values in {0, 255}.
    """
    error_map = np.squeeze(error_map)
    if error_map.ndim != 2:
        raise ValueError(f"error_map must be 2D [H, W], got shape {error_map.shape}")

    # 1. Normalize error map to uint8 [0, 255]
    min_v = float(np.min(error_map))
    max_v = float(np.max(error_map))
    if max_v > min_v:
        norm_map = ((error_map - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        return np.zeros_like(error_map, dtype=np.uint8)

    # 2. Gaussian smoothing to suppress sensor / optical high-frequency noise
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(norm_map, (blur_kernel, blur_kernel), 0)
    else:
        blurred = norm_map

    # 3. Binarization
    if method == "otsu":
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "percentile":
        cutoff = float(np.percentile(blurred, percentile))
        _, mask = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)
    else:
        # Fixed ratio thresholding
        thresh_val = int(threshold_ratio * 255)
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # 4. Morphological filtering (closing to fill holes, dilation to link components)
    if morph_kernel_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        # Morphological close: removes small dark holes inside foreground
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        if dilation_iters > 0:
            mask = cv2.dilate(mask, k, iterations=dilation_iters)

    return mask


def merge_overlapping_boxes(bboxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """
    Merge heavily overlapping or nested bounding boxes.

    Args:
        bboxes: List of (x, y, w, h) bounding boxes.
        iou_thresh: Intersection-over-Union threshold for merging.

    Returns:
        Merged list of (x, y, w, h) bounding boxes.
    """
    if not bboxes:
        return []

    # Convert to [x1, y1, x2, y2]
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bboxes], dtype=np.float32)
    merged = []
    visited = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if visited[i]:
            continue

        current_box = boxes[i].copy()
        visited[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if visited[j]:
                    continue

                # Compute IoU
                b = boxes[j]
                xx1 = max(current_box[0], b[0])
                yy1 = max(current_box[1], b[1])
                xx2 = min(current_box[2], b[2])
                yy2 = min(current_box[3], b[3])

                w_inter = max(0.0, xx2 - xx1)
                h_inter = max(0.0, yy2 - yy1)
                inter = w_inter * h_inter

                area_cur = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                union = area_cur + area_b - inter

                iou = inter / union if union > 0 else 0

                # Also merge if one box is mostly contained inside the other
                containment = inter / min(area_cur, area_b) if min(area_cur, area_b) > 0 else 0

                if iou > iou_thresh or containment > 0.6:
                    current_box[0] = min(current_box[0], b[0])
                    current_box[1] = min(current_box[1], b[1])
                    current_box[2] = max(current_box[2], b[2])
                    current_box[3] = max(current_box[3], b[3])
                    visited[j] = True
                    changed = True

        merged.append((
            int(current_box[0]),
            int(current_box[1]),
            int(current_box[2] - current_box[0]),
            int(current_box[3] - current_box[1])
        ))

    return merged


def extract_bboxes_from_mask(
    mask: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None,
    min_area: int = 150,
    max_area: Optional[int] = None,
    merge_overlapping: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from a binary mask and optionally scale to original resolution.

    Args:
        mask: 2D binary uint8 mask [H, W].
        orig_shape: Tuple of (orig_height, orig_width). If None, uses mask dimensions.
        min_area: Minimum box area in original resolution to filter minor noise.
        max_area: Maximum box area (e.g. 80% of frame) to filter global lighting flashes.
        merge_overlapping: Whether to merge overlapping/touching bounding boxes.

    Returns:
        List of bounding boxes in format: [(x, y, w, h), ...]
    """
    model_h, model_w = mask.shape[:2]
    if orig_shape is not None:
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / float(model_w)
        scale_y = orig_h / float(model_h)
    else:
        orig_h, orig_w = model_h, model_w
        scale_x = scale_y = 1.0

    if max_area is None:
        max_area = int(orig_h * orig_w * 0.85)

    # Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Scale to original dimensions
        box_x = int(x * scale_x)
        box_y = int(y * scale_y)
        box_w = int(w * scale_x)
        box_h = int(h * scale_y)

        area = box_w * box_h
        if min_area <= area <= max_area:
            raw_boxes.append((box_x, box_y, box_w, box_h))

    if merge_overlapping and len(raw_boxes) > 1:
        return merge_overlapping_boxes(raw_boxes)

    return raw_boxes


def extract_bboxes_from_error_map(
    error_map: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None,
    method: str = "otsu",
    threshold_ratio: float = 0.5,
    blur_kernel: int = 7,
    morph_kernel_size: int = 5,
    dilation_iters: int = 2,
    min_area: int = 200,
    max_area: Optional[int] = None,
    merge_overlapping: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    End-to-end extraction from a 2D error map to bounding boxes and binary mask.

    Returns:
        (bboxes, binary_mask)
    """
    mask = error_map_to_mask(
        error_map=error_map,
        method=method,
        threshold_ratio=threshold_ratio,
        blur_kernel=blur_kernel,
        morph_kernel_size=morph_kernel_size,
        dilation_iters=dilation_iters,
    )

    bboxes = extract_bboxes_from_mask(
        mask=mask,
        orig_shape=orig_shape,
        min_area=min_area,
        max_area=max_area,
        merge_overlapping=merge_overlapping,
    )

    return bboxes, mask


def draw_bboxes(
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    label: str = "",
    color: Tuple[int, int, int] = (0, 0, 255),  # BGR Red
    thickness: int = 1,
    show_label: bool = False,
    scores: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Draw slim, crisp bounding boxes on an image frame without text badges.

    Args:
        frame: OpenCV image frame [H, W, 3] in BGR format.
        bboxes: List of (x, y, w, h) bounding boxes.
        label: Optional text string for badge (only drawn if show_label=True).
        color: Box outline BGR color tuple.
        thickness: Line thickness in pixels (default 1 for slim borders).
        show_label: Whether to render label text tag (default False).
        scores: Optional float score per box.

    Returns:
        Annotated frame copy.
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]

    for idx, (bx, by, bw, bh) in enumerate(bboxes):
        x1 = max(0, min(w - 1, bx))
        y1 = max(0, min(h - 1, by))
        x2 = max(0, min(w - 1, bx + bw))
        y2 = max(0, min(h - 1, by + bh))

        # Draw slim, clean rectangle with anti-aliasing
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

        if show_label and label:
            tag_text = label
            if scores is not None and idx < len(scores):
                tag_text += f": {scores[idx]:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)

            tag_y1 = max(0, y1 - th - baseline - 4)
            tag_y2 = y1
            tag_x2 = min(w - 1, x1 + tw + 8)

            cv2.rectangle(out_frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                out_frame,
                tag_text,
                (x1 + 4, tag_y2 - baseline),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return out_frame


def overlay_heatmap(
    frame: np.ndarray,
    error_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a thermal anomaly error heatmap over the original frame.

    Args:
        frame: Original image frame [H, W, 3] in BGR format.
        error_map: 2D error map float array [model_h, model_w].
        alpha: Blending weight for heatmap (0.0 to 1.0).
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).

    Returns:
        Blended BGR image frame.
    """
    orig_h, orig_w = frame.shape[:2]
    error_map = np.squeeze(error_map)

    # Resize error map to match original frame
    resized_err = cv2.resize(error_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 255]
    min_v = float(np.min(resized_err))
    max_v = float(np.max(resized_err))
    if max_v > min_v:
        norm_map = ((resized_err - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        norm_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Apply colormap
    color_heat = cv2.applyColorMap(norm_map, colormap)

    # Blend with original
    blended = cv2.addWeighted(frame, 1.0 - alpha, color_heat, alpha, 0)
    return blended


class AnomalyBoundingBoxDetector:
    """
    Modular engine for extracting and annotating bounding boxes
    from ASTNet error maps.
    """

    def __init__(
        self,
        method: str = "otsu",
        threshold_ratio: float = 0.5,
        min_area: int = 200,
        blur_kernel: int = 7,
        morph_kernel_size: int = 5,
        dilation_iters: int = 2,
        label: str = "ANOMALY",
        color: Tuple[int, int, int] = (0, 0, 255),
    ):
        self.method = method
        self.threshold_ratio = threshold_ratio
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.morph_kernel_size = morph_kernel_size
        self.dilation_iters = dilation_iters
        self.label = label
        self.color = color

    def process(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        orig_frame: Optional[np.ndarray] = None,
        draw_boxes: bool = True,
        blend_heat: bool = False,
    ) -> Dict:
        """
        Process a single predicted and target frame to extract bounding boxes.

        Returns:
            dict containing:
                - 'bboxes': List of (x, y, w, h)
                - 'error_map': 2D float error map
                - 'mask': 2D binary uint8 mask
                - 'annotated_frame': Rendered frame (if orig_frame provided)
        """
        orig_shape = orig_frame.shape[:2] if orig_frame is not None else None

        error_map = compute_spatial_error_map(pred, target)
        bboxes, mask = extract_bboxes_from_error_map(
            error_map=error_map,
            orig_shape=orig_shape,
            method=self.method,
            threshold_ratio=self.threshold_ratio,
            blur_kernel=self.blur_kernel,
            morph_kernel_size=self.morph_kernel_size,
            dilation_iters=self.dilation_iters,
            min_area=self.min_area,
        )

        annotated = None
        if orig_frame is not None:
            annotated = orig_frame.copy()
            if blend_heat:
                annotated = overlay_heatmap(annotated, error_map)
            if draw_boxes and bboxes:
                annotated = draw_bboxes(
                    annotated,
                    bboxes,
                    label=self.label,
                    color=self.color,
                )

        return {
            "bboxes": bboxes,
            "error_map": error_map,
            "mask": mask,
            "annotated_frame": annotated,
        }
