"""
Prediction Error Heatmap Generator Module
------------------------------------------
Location: d:/anomaly-detection/src/prediction_error_heatmap.py

Provides utilities and an end-to-end renderer for transforming ASTNet
spatiotemporal reconstruction error maps into thermal JET heatmaps,
generating synchronized video streams for the Model Output interface.
"""

import os
import sys
import tempfile
import subprocess
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_prediction_error_map(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    method: str = "mse",
) -> np.ndarray:
    """
    Compute 2D spatial reconstruction error map between predicted frame and ground-truth frame.

    Args:
        pred: Predicted frame tensor [B, 3, H, W] or [3, H, W] or numpy array.
        target: Target frame tensor with matching shape to pred.
        method: 'mse' (squared difference) or 'l1' (absolute difference).

    Returns:
        error_map: 2D numpy array [H, W] (or [B, H, W] if batch).
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        if pred.min() < 0.0 or target.min() < 0.0:
            p_01 = (pred + 1.0) * 0.5
            t_01 = (target + 1.0) * 0.5
        else:
            p_01 = pred
            t_01 = target

        dim = 1 if p_01.dim() == 4 else 0
        if method.lower() == "l1":
            diff = torch.mean(torch.abs(p_01 - t_01), dim=dim)
        else:
            diff = torch.mean((p_01 - t_01) ** 2, dim=dim)

        res = diff.detach().cpu().numpy()
        if res.ndim == 3 and res.shape[0] == 1:
            res = res.squeeze(0)
        return res

    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)

    if p.min() < 0.0 or t.min() < 0.0:
        p = (p + 1.0) * 0.5
        t = (t + 1.0) * 0.5

    if p.ndim == 3:
        axis = 0 if p.shape[0] in (1, 3) else 2
    elif p.ndim == 4:
        axis = 1
    else:
        raise ValueError(f"Unsupported array dimensions: {p.ndim}")

    if method.lower() == "l1":
        res = np.mean(np.abs(p - t), axis=axis)
    else:
        res = np.mean((p - t) ** 2, axis=axis)

    if res.ndim == 3 and res.shape[0] == 1:
        res = res.squeeze(0)
    return res


def error_map_to_jet_heatmap(
    error_map: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: int = cv2.COLORMAP_JET,
    blur_kernel: int = 5,
) -> np.ndarray:
    """
    Transform a 2D float error map into an aesthetically pleasing thermal JET heatmap.

    Args:
        error_map: 2D float error map [H, W].
        orig_shape: Optional target (height, width) tuple to resize the output.
        vmin: Minimum value for normalization. If None, uses local min.
        vmax: Maximum value for normalization. If None, uses local max.
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).
        blur_kernel: Size of Gaussian blur kernel for smooth thermal gradients.

    Returns:
        heatmap: 3-channel BGR uint8 image of shape (orig_h, orig_w, 3).
    """
    err = np.squeeze(error_map).astype(np.float32)
    if err.ndim != 2:
        raise ValueError(f"error_map must be 2D [H, W], got shape {err.shape}")

    # Determine normalization bounds
    min_val = float(vmin) if vmin is not None else float(np.min(err))
    max_val = float(vmax) if vmax is not None else float(np.max(err))

    if max_val > min_val:
        norm = np.clip((err - min_val) / (max_val - min_val), 0.0, 1.0)
        norm_uint8 = (norm * 255.0).astype(np.uint8)
    else:
        norm_uint8 = np.zeros_like(err, dtype=np.uint8)

    # Apply Gaussian smoothing for continuous gradient flow
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        norm_uint8 = cv2.GaussianBlur(norm_uint8, (k, k), 0)

    # Resize to target resolution if specified
    if orig_shape is not None:
        target_h, target_w = orig_shape[:2]
        if (norm_uint8.shape[0], norm_uint8.shape[1]) != (target_h, target_w):
            norm_uint8 = cv2.resize(norm_uint8, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Apply JET colormap: low values -> deep blue, high values -> cyan/yellow/red
    heat_bgr = cv2.applyColorMap(norm_uint8, colormap)
    return heat_bgr


def overlay_heatmap_on_frame(
    frame: np.ndarray,
    error_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    blur_kernel: int = 5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Blend a thermal JET error heatmap over an original video frame.

    Args:
        frame: Original image frame [H, W, 3] in BGR format.
        error_map: 2D float error map [model_h, model_w].
        alpha: Blending weight for heatmap (0.0 = original frame, 1.0 = heatmap).
        colormap: OpenCV colormap constant.
        blur_kernel: Gaussian blur kernel size.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.

    Returns:
        Blended BGR image frame.
    """
    orig_h, orig_w = frame.shape[:2]
    heat_bgr = error_map_to_jet_heatmap(
        error_map=error_map,
        orig_shape=(orig_h, orig_w),
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        blur_kernel=blur_kernel,
    )
    return cv2.addWeighted(frame, 1.0 - alpha, heat_bgr, alpha, 0)


def render_prediction_error_video(
    video_path: str,
    fps: float,
    frame_spatial_errors: Dict[int, np.ndarray],
    output_path: Optional[str] = None,
    colormap: int = cv2.COLORMAP_JET,
    blur_kernel: int = 0,
    blend_with_raw: bool = False,
    alpha: float = 0.5,
    normalization: str = "dynamic",
) -> str:
    """
    Render a synchronized Prediction Error Heatmap video from video frames
    and spatial error maps, converting to web-compatible H.264 MP4.

    Args:
        video_path: Path to the original input video.
        fps: Playback frame rate.
        frame_spatial_errors: Dict mapping frame index to 2D numpy error map.
        output_path: Optional output video path. If None, saves in temp directory.
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).
        blur_kernel: Gaussian blur kernel size for gradient smoothing (0 = sharp).
        blend_with_raw: If True, blends heatmap with raw frame; if False, generates standalone JET heatmap.
        alpha: Blending factor when blend_with_raw=True.
        normalization: 'dynamic' (per-frame contrast, matching reference) or 'global'.

    Returns:
        out_video_path: Absolute path to the web-compatible H.264 MP4 video.
    """
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        cap.release()
        return video_path

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(temp_dir, f"{base_name}_error_heatmap.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    # Calculate global normalization bounds across all frames if requested
    global_vmin, global_vmax = None, None
    if normalization == "global" and frame_spatial_errors:
        all_mins = [float(np.min(m)) for m in frame_spatial_errors.values()]
        all_maxs = [float(np.max(m)) for m in frame_spatial_errors.values()]
        global_vmin = float(np.min(all_mins))
        global_vmax = float(np.max(all_maxs))
        if global_vmax <= global_vmin:
            global_vmin, global_vmax = None, None

    # Find earliest available error map to handle initial warmup frames
    first_available_error = None
    if frame_spatial_errors:
        sorted_keys = sorted(frame_spatial_errors.keys())
        first_available_error = frame_spatial_errors[sorted_keys[0]]

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Select corresponding error map or fallback to warmup frame
        if f_idx in frame_spatial_errors:
            err_map = frame_spatial_errors[f_idx]
        elif first_available_error is not None:
            err_map = first_available_error
        else:
            err_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        # Normalization bounds for this frame
        if normalization == "dynamic":
            f_min = float(np.min(err_map))
            f_max = float(np.max(err_map))
            vmin_f, vmax_f = f_min, f_max
        else:
            vmin_f, vmax_f = global_vmin, global_vmax

        if blend_with_raw:
            out_frame = overlay_heatmap_on_frame(
                frame=frame,
                error_map=err_map,
                alpha=alpha,
                colormap=colormap,
                blur_kernel=blur_kernel,
                vmin=vmin_f,
                vmax=vmax_f,
            )
        else:
            out_frame = error_map_to_jet_heatmap(
                error_map=err_map,
                orig_shape=(orig_h, orig_w),
                vmin=vmin_f,
                vmax=vmax_f,
                colormap=colormap,
                blur_kernel=blur_kernel,
            )

        writer.write(out_frame)
        f_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 for HTML5 web browser compatibility
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = output_path + ".h264.mp4"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            temp_h264,
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and os.path.exists(temp_h264):
            os.replace(temp_h264, output_path)
    except Exception:
        pass

    return output_path


class PredictionErrorHeatmapRenderer:
    """
    Modular engine for generating prediction error heatmaps and videos.
    """

    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        blur_kernel: int = 5,
        blend_with_raw: bool = False,
        alpha: float = 0.5,
        normalization: str = "global",
    ):
        self.colormap = colormap
        self.blur_kernel = blur_kernel
        self.blend_with_raw = blend_with_raw
        self.alpha = alpha
        self.normalization = normalization

    def render_video(
        self,
        video_path: str,
        fps: float,
        frame_spatial_errors: Dict[int, np.ndarray],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render the heatmap video.
        """
        return render_prediction_error_video(
            video_path=video_path,
            fps=fps,
            frame_spatial_errors=frame_spatial_errors,
            output_path=output_path,
            colormap=self.colormap,
            blur_kernel=self.blur_kernel,
            blend_with_raw=self.blend_with_raw,
            alpha=self.alpha,
            normalization=self.normalization,
        )
