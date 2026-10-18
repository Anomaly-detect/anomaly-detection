"""
ASTNet Anomaly Detection Inference Engine
-----------------------------------------
Location: d:/anomaly-detection/src/detector.py
Provides a clean, modular adapter for running video anomaly detection
using the ASTNet (Attention-based Spatio-Temporal Network) architecture.
"""

import os
import sys
import time
import tempfile
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ASTNetDetector:
    """
    Adapter class for loading ASTNet models and running anomaly detection on videos.
    """

    def __init__(
        self,
        config_path: str = "config/ped2_wresnet.yaml",
        checkpoint_path: str = "checkpoints/ped2.pth",
        device: str = None,
    ):
        """
        Initialize the ASTNet model with config and pretrained weights.
        """
        # Resolve absolute paths
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Resolve device
        if device is None:
            self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)

        # Load config
        from config.defaults import _C as default_cfg

        self.config = default_cfg.clone()
        self.config.defrost()
        if os.path.exists(config_path):
            self.config.merge_from_file(config_path)
        else:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        self.config.freeze()

        # Architecture selection
        dataset_name = self.config.DATASET.DATASET
        if dataset_name == "ped2":
            from models.wresnet1024_cattn_tsm import ASTNet
        else:
            from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet

        # Initialize network (pretrained=False because we load complete checkpoint)
        self.model = ASTNet(self.config, pretrained=False)

        # Load checkpoint weights
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Clean state dict keys (strip 'module.' prefix if saved with DataParallel)
        cleaned_sd = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        self.model.load_state_dict(cleaned_sd)

        # Transfer model to target device
        self.model.to(self.device)
        self.model.eval()

        # Input configuration
        self.encoded_frames = getattr(self.config.MODEL, "ENCODED_FRAMES", 4)
        # Ped2 config: [224, 288] -> height 224, width 288
        self.img_h = int(self.config.MODEL.IMAGE_SIZE[0])
        self.img_w = int(self.config.MODEL.IMAGE_SIZE[1])

    def get_model_info(self) -> dict:
        """
        Return structured metadata about the loaded model and environment.
        """
        return {
            "config_file": os.path.basename(self.config_path),
            "config_path": self.config_path,
            "dataset": getattr(self.config.DATASET, "DATASET", "ped2"),
            "backbone": getattr(self.config.MODEL, "NAME", "wrn38"),
            "checkpoint": os.path.basename(self.checkpoint_path),
            "device": str(self.device).upper(),
            "input_resolution": f"{self.img_h}x{self.img_w}",
            "encoded_frames": self.encoded_frames,
        }

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single OpenCV BGR frame:
        1. Resize to target dimension (img_w, img_h)
        2. Convert BGR to RGB
        3. Normalize to [-1.0, 1.0]
        4. Transpose to (C, H, W) and convert to float tensor
        """
        resized = cv2.resize(frame_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [-1.0, 1.0]
        normalized = (rgb.astype(np.float32) / 127.5) - 1.0
        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        return tensor

    def predict_video(
        self,
        video_path: str,
        threshold: float = 0.5,
        batch_size: int = 4,
        progress_callback=None,
    ) -> dict:
        """
        Run spatiotemporal anomaly detection on the specified video file.

        Args:
            video_path: Path to the target video file.
            threshold: Decision threshold for anomaly classification (0.0 to 1.0).
            batch_size: Mini-batch size for sliding window inference.
            progress_callback: Optional callable(percent_float, status_str).

        Returns:
            dict with processed_video_path, timestamps, scores, peak_score, etc.
        """
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # 1. Read video metadata and decode frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        raw_tensors = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_tensors.append(self.preprocess_frame(frame))
        cap.release()

        actual_total_frames = len(raw_tensors)
        if actual_total_frames == 0:
            raise ValueError(f"Could not read any frames from: {video_path}")

        duration = actual_total_frames / fps
        timestamps = [round(i / fps, 2) for i in range(actual_total_frames)]

        fp = self.encoded_frames  # Typically 4 frames
        if actual_total_frames <= fp:
            # Video too short to form a prediction window
            scores = [0.0] * actual_total_frames
            return {
                "processed_video_path": video_path,
                "annotated_video_path": video_path,
                "heatmap_video_path": video_path,
                "timestamps": timestamps,
                "scores": scores,
                "peak_score": 0.0,
                "is_anomaly": False,
                "fps": fps,
                "total_frames": actual_total_frames,
                "duration": duration,
                "inference_time": time.time() - start_time,
            }

        # 2. Build sliding window sequences
        # Each window i: inputs = [f_i, f_{i+1}, ..., f_{i+fp-1}], target = f_{i+fp}
        num_windows = actual_total_frames - fp
        window_indices = list(range(num_windows))

        psnr_results = []
        total_batches = (num_windows + batch_size - 1) // batch_size

        # 3. Batch inference on GPU / CPU
        frame_spatial_errors = {}
        frame_spatial_l1 = {}
        with torch.no_grad():
            for b_idx in range(total_batches):
                b_start = b_idx * batch_size
                b_end = min(b_start + batch_size, num_windows)

                # Prepare batch inputs: list of fp tensors, each shape [B, 3, H, W]
                batch_inputs = []
                for step in range(fp):
                    step_tensors = [raw_tensors[i + step] for i in range(b_start, b_end)]
                    batch_inputs.append(torch.stack(step_tensors).to(self.device, non_blocking=True))

                # Batch target: shape [B, 3, H, W]
                batch_target = torch.stack(
                    [raw_tensors[i + fp] for i in range(b_start, b_end)]
                ).to(self.device, non_blocking=True)

                # Forward pass: model generates predicted next frame on GPU
                output = self.model(batch_inputs)

                # Compute MSE per item in batch directly on GPU
                # Map from [-1.0, 1.0] to [0.0, 1.0]
                pred_01 = (output + 1.0) * 0.5
                tgt_01 = (batch_target + 1.0) * 0.5

                # Mean squared error over (C, H, W) on GPU
                mse = torch.mean((pred_01 - tgt_01) ** 2, dim=(1, 2, 3))
                # Compute PSNR (Park et al. definition) natively on GPU
                batch_psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))
                psnr_results.extend(batch_psnr.cpu().tolist())

                # 2D spatial error map [B, H, W] for bounding box extraction (MSE) and heatmap (L1)
                batch_spatial_err = torch.mean((pred_01 - tgt_01) ** 2, dim=1).detach().cpu().numpy()
                batch_spatial_l1 = torch.mean(torch.abs(pred_01 - tgt_01), dim=1).detach().cpu().numpy()
                for idx in range(b_end - b_start):
                    frame_idx = b_start + idx + fp
                    frame_spatial_errors[frame_idx] = batch_spatial_err[idx]
                    frame_spatial_l1[frame_idx] = batch_spatial_l1[idx]

                # Throttle progress callback to prevent CPU/WebSocket bottlenecks
                if progress_callback:
                    step = max(1, total_batches // 40)
                    if (b_idx + 1) % step == 0 or (b_idx + 1) == total_batches:
                        try:
                            progress_callback(b_end, num_windows)
                        except TypeError:
                            progress_callback(f"{b_end}/{num_windows}")

        # 4. Map window PSNRs back to all frames
        # The first fp frames inherit the PSNR of frame fp
        full_psnr = [psnr_results[0]] * fp + psnr_results

        # 5. Convert PSNR to Anomaly Score in range [0.0, 1.0]
        # Low PSNR = Higher reconstruction error = Anomaly!
        psnr_arr = np.array(full_psnr, dtype=np.float32)
        min_p = float(np.min(psnr_arr))
        max_p = float(np.max(psnr_arr))

        if max_p > min_p:
            # Regularity score: (PSNR - min) / (max - min) (1.0 = Normal, 0.0 = Abnormal)
            # Anomaly score: 1.0 - Regularity score (1.0 = Abnormal, 0.0 = Normal)
            raw_scores = 1.0 - ((psnr_arr - min_p) / (max_p - min_p))
        else:
            raw_scores = np.full_like(psnr_arr, 0.1)

        # 6. Temporal smoothing (Moving average filter with window size 5)
        # Eliminates single-frame jitter and provides clean visual curves
        kernel_size = min(5, len(raw_scores))
        if kernel_size > 1:
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            smooth_scores = np.convolve(raw_scores, kernel, mode="same")
        else:
            smooth_scores = raw_scores

        scores = [round(float(s), 4) for s in np.clip(smooth_scores, 0.0, 1.0)]

        peak_score = float(np.max(scores)) if scores else 0.0
        inference_duration = time.time() - start_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. Render bounding boxes onto raw video frames for anomaly events
        annotated_video_path = self.render_annotated_video(
            video_path=video_path,
            fps=fps,
            scores=scores,
            threshold=threshold,
            frame_spatial_errors=frame_spatial_errors,
        )

        # 8. Render Prediction Error Heatmap video for Model Output screen
        from src.prediction_error_heatmap import render_prediction_error_video
        heatmap_video_path = render_prediction_error_video(
            video_path=video_path,
            fps=fps,
            frame_spatial_errors=frame_spatial_l1,
            blur_kernel=0,
            normalization="dynamic",
        )

        return {
            "annotated_video_path": annotated_video_path,
            "processed_video_path": annotated_video_path,
            "heatmap_video_path": heatmap_video_path,
            "timestamps": timestamps,
            "scores": scores,
            "peak_score": peak_score,
            "is_anomaly": peak_score >= threshold,
            "fps": fps,
            "total_frames": actual_total_frames,
            "duration": duration,
            "inference_time": inference_duration,
        }


    def render_annotated_video(
        self,
        video_path: str,
        fps: float,
        scores: list,
        threshold: float,
        frame_spatial_errors: dict,
    ) -> str:
        """
        Draw bounding boxes on anomalous frames and export an H.264 MP4 video.
        """
        from src.bounding_box import extract_bboxes_from_error_map, draw_bboxes

        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w <= 0 or orig_h <= 0:
            cap.release()
            return video_path

        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_video_path = os.path.join(temp_dir, f"{base_name}_annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_w, orig_h))

        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            score = scores[f_idx] if f_idx < len(scores) else 0.0
            if score >= threshold and f_idx in frame_spatial_errors:
                err_map = frame_spatial_errors[f_idx]
                bboxes, _ = extract_bboxes_from_error_map(
                    error_map=err_map,
                    orig_shape=(orig_h, orig_w),
                    min_area=150,
                )
                if bboxes:
                    frame = draw_bboxes(
                        frame,
                        bboxes,
                        color=(0, 0, 255),
                        thickness=1,
                        show_label=False,
                    )

            writer.write(frame)
            f_idx += 1

        cap.release()
        writer.release()

        # Re-encode to H.264 for HTML5 web browser compatibility
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_h264 = out_video_path + ".h264.mp4"
            cmd = [
                ffmpeg_exe, "-y",
                "-i", out_video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "veryfast",
                temp_h264,
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and os.path.exists(temp_h264):
                os.replace(temp_h264, out_video_path)
        except Exception:
            pass

        return out_video_path


