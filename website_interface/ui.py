r"""
Video Anomaly Detection Studio - UI Module
------------------------------------------
Location: D:\anomaly-detection\website_interface\ui.py
Contains all user interface components, CSS styles, and synchronized dual player logic.
"""

import os
import sys
import time
import tempfile
import base64
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is available
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==============================================================================
# 1. Model Inference Pipeline Hook (ASTNet - ped2_wresnet)
# ==============================================================================
PED2_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "ped2_wresnet.yaml")
PED2_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ped2.pth")


@st.cache_resource(show_spinner=False)
def get_astnet_detector(
    config_path: str = PED2_CONFIG_PATH,
    checkpoint_path: str = PED2_CHECKPOINT_PATH,
    reload_key: str = "v6_triple_predicted_frame",
):
    """
    Cached detector singleton instance so model weights stay in GPU VRAM across reruns.
    Strictly locked to the ped2_wresnet configuration.
    """
    import importlib
    import src.detector
    import src.prediction_error_heatmap
    import src.predicted_frame
    importlib.reload(src.detector)
    importlib.reload(src.prediction_error_heatmap)
    importlib.reload(src.predicted_frame)
    from src.detector import ASTNetDetector

    return ASTNetDetector(config_path=config_path, checkpoint_path=checkpoint_path)


def run_anomaly_detection(video_path: str, threshold: float = 0.5, progress_callback=None):
    """
    Deep learning anomaly detection model execution using ASTNet with ped2_wresnet config.

    Args:
        video_path: Path to the uploaded video file.
        threshold: Anomaly score decision threshold (0.0 to 1.0).
        progress_callback: Optional callable for live progress bar updates.

    Returns:
        dict containing processed_video_path, timestamps, scores, peak_score, etc.
    """
    detector = get_astnet_detector(
        config_path=PED2_CONFIG_PATH,
        checkpoint_path=PED2_CHECKPOINT_PATH,
        reload_key="v6_triple_predicted_frame",
    )
    return detector.predict_video(
        video_path=video_path,
        threshold=threshold,
        batch_size=4,
        progress_callback=progress_callback,
    )



# ==============================================================================
# 2. Modern Blue & White Styling (CSS)
# ==============================================================================
def apply_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Background & Container */
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Header Center */
        .app-header-center {
            text-align: center;
            margin-top: 0.5rem;
            margin-bottom: 2rem;
        }

        .app-header-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.03em;
            margin: 0;
            line-height: 1.2;
        }

        .app-header-subtitle {
            font-size: 0.95rem;
            color: #64748B;
            margin-top: 0.4rem;
            font-weight: 500;
        }

        .model-badge-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 0.85rem;
            margin-bottom: 0.25rem;
        }

        .model-badge-item {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 9999px;
            padding: 0.28rem 0.75rem;
            font-size: 0.775rem;
            color: #475569;
            font-weight: 600;
        }

        .model-badge-item.active {
            background: #EFF6FF;
            border-color: #BFDBFE;
            color: #1D4ED8;
        }

        /* Unified Card System using Streamlit container border wrapper */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 18px !important;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05) !important;
            margin-bottom: 1.75rem !important;
            padding: 0.5rem !important;
        }

        /* Remove outermost wrapper border, background, shadow, and padding inside columns (such as KPI metric cards and controls) */
        [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stColumn"] > div,
        [data-testid="column"] > div {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-bottom: 0 !important;
        }

        .card-top-header {
            margin-bottom: 1rem;
            padding: 0.25rem 0.25rem 0 0.25rem;
        }

        .card-main-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #0F172A;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            letter-spacing: -0.01em;
        }

        .card-main-desc {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.25rem;
            font-weight: 500;
        }

        .card-divider {
            height: 1px;
            background: #F1F5F9;
            margin: 1.25rem 0 1rem 0;
        }

        /* File Uploader styling inside Unified Card */
        [data-testid="stFileUploader"] {
            background: transparent !important;
            padding: 0 !important;
            border: none !important;
            width: 100% !important;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploader"] section {
            background: #F8FAFC !important;
            border: 2px dashed #93C5FD !important;
            border-radius: 14px !important;
            height: 220px !important;
            min-height: 220px !important;
            max-height: 220px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            gap: 0.75rem !important;
            padding: 0 !important;
            margin: 0 !important;
            box-sizing: border-box !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover,
        [data-testid="stFileUploadDropzone"]:hover,
        [data-testid="stFileUploader"] section:hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
        }

        /* Wrap span around the button */
        [data-testid="stFileUploaderDropzone"] > span,
        [data-testid="stFileUploadDropzone"] > span,
        [data-testid="stFileUploader"] section > span {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
            flex: 0 0 auto !important;
        }

        /* Center the upload button inside the dropzone */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] section button,
        [data-testid="stFileUploader"] button {
            margin: 0 auto !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            padding: 0.65rem 1.8rem !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
            border: 1px solid #BFDBFE !important;
            color: #1D4ED8 !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            flex: 0 0 auto !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploadDropzone"] button:hover,
        [data-testid="stFileUploader"] section button:hover,
        [data-testid="stFileUploader"] button:hover {
            background: #EFF6FF !important;
            border-color: #2563EB !important;
            color: #1D4ED8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2) !important;
        }

        /* Cancel out Streamlit internal flex: 1 on instructions so it doesn't push the button */
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploader"] section > div {
            flex: 0 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
            gap: 0.25rem !important;
            align-self: center !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] * {
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }

        .file-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 8px;
            padding: 0.45rem 0.9rem;
            font-size: 0.85rem;
            color: #1E40AF;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        /* KPI Metric Cards */
        .metric-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .metric-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
            border-color: #BFDBFE;
        }

        .metric-header {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.4rem;
        }

        .metric-number {
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }

        .metric-subtext {
            font-size: 0.75rem;
            color: #94A3B8;
            font-weight: 500;
            margin-top: 0.35rem;
        }

        /* Buttons */
        div.stButton > button,
        div.stButton > button:first-child {
            box-shadow: none !important;
            filter: none !important;
        }

        div.stButton > button[data-testid="stBaseButton-primary"],
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.75rem !important;
            box-shadow: none !important;
            filter: none !important;
            transition: all 0.2s ease-in-out !important;
        }

        div.stButton > button[data-testid="stBaseButton-primary"]:hover,
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%) !important;
            box-shadow: none !important;
            filter: none !important;
            transform: translateY(-1px) !important;
        }

        div.stButton > button[data-testid="stBaseButton-secondary"],
        div.stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #1E40AF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            filter: none !important;
        }

        div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
        div.stButton > button[kind="secondary"]:hover {
            background: #F1F5F9 !important;
            border-color: #94A3B8 !important;
            box-shadow: none !important;
            filter: none !important;
        }

        div.stButton > button:focus,
        div.stButton > button:active,
        div.stButton > button:focus-visible {
            box-shadow: none !important;
            filter: none !important;
            outline: none !important;
        }

        /* Material Icons inside Buttons */
        div.stButton > button [data-testid="stIconMaterial"] {
            font-size: 1.25rem !important;
            line-height: 1 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div.stButton > button[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"],
        div.stButton > button[kind="primary"] [data-testid="stIconMaterial"] {
            color: #FFFFFF !important;
        }

        div.stButton > button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"],
        div.stButton > button[kind="secondary"] [data-testid="stIconMaterial"] {
            color: #1E40AF !important;
        }

        /* Slider */
        div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background-color: #1D4ED8 !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
        }
        div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
            color: #1D4ED8 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
            background: #2563EB !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. Synchronized Dual Video Player Component
# ==============================================================================
def ensure_h264_compatible(video_path: str) -> str:
    """
    Ensures video is encoded in standard H.264 (libx264/yuv420p) with faststart
    so that Google Chrome, Edge, and Firefox HTML5 video tags can play it cleanly
    without black screens or unsupported codec errors.
    """
    if not video_path or not os.path.exists(video_path):
        return video_path
    try:
        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        base_name, _ = os.path.splitext(video_path)
        h264_path = f"{base_name}_web.mp4"

        # If converted copy already exists and is newer than source, reuse it
        if os.path.exists(h264_path) and os.path.getmtime(h264_path) >= os.path.getmtime(video_path):
            return h264_path

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            h264_path,
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and os.path.exists(h264_path):
            return h264_path
    except Exception:
        pass
    return video_path


def render_synchronized_dual_video(video_left_path: str, video_right_path: str, fps: float = 30.0):
    """
    Renders an HTML5 dual video player with unified synchronized Play, Pause,
    Seek, Frame-Stepping, and Speed controls keeping both streams in lockstep:
    1. Raw Video (Bounding Box)
    2. Prediction Error Heatmap
    """
    v_left_play = ensure_h264_compatible(video_left_path)
    v_right_play = ensure_h264_compatible(video_right_path)

    with open(v_left_play, "rb") as f:
        v_left_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(v_right_play, "rb") as f:
        v_right_b64 = base64.b64encode(f.read()).decode("utf-8")

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
      body {{ background: transparent; padding: 0; overflow: hidden; }}
      
      .dual-player-box {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
      }}

      .videos-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 0.85rem;
      }}

      @media (max-width: 992px) {{
        .videos-grid {{
          grid-template-columns: 1fr;
        }}
      }}

      .video-panel {{
        background: #0B0F19;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      }}

      .video-panel-header {{
        background: #0F172A;
        color: #F8FAFC;
        padding: 0.55rem 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }}

      .video-panel-title {{
        font-size: 0.85rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #FFFFFF;
      }}

      .badge-tag {{
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }}

      .badge-blue {{ background: #1D4ED8; color: #FFFFFF; }}
      .badge-red {{ background: #DC2626; color: #FFFFFF; }}

      video {{
        width: 100%;
        height: 310px;
        object-fit: contain;
        background: #000000;
        display: block;
        cursor: pointer;
      }}

      /* Unified Controller Bar */
      .sync-controller {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
      }}

      .btn-play {{
        background: linear-gradient(135deg, #1D4ED8, #2563EB);
        color: #FFFFFF;
        border: none;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
        transition: transform 0.1s ease, background 0.15s ease;
      }}

      .btn-play:hover {{ transform: scale(1.05); }}

      .btn-step {{
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #334155;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
      }}

      .btn-step:hover {{
        background: #EFF6FF;
        border-color: #93C5FD;
        color: #1D4ED8;
      }}

      .time-display {{
        font-size: 0.825rem;
        font-weight: 700;
        color: #334155;
        font-variant-numeric: tabular-nums;
        min-width: 95px;
      }}

      .scrubber-wrap {{
        flex: 1;
        display: flex;
        align-items: center;
      }}

      .scrubber {{
        width: 100%;
        accent-color: #2563EB;
        cursor: pointer;
        height: 6px;
      }}

      .speed-select {{
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.25rem 0.45rem;
        font-size: 0.775rem;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
      }}
    </style>
    </head>
    <body>
      <div class="dual-player-box">
        <div class="videos-grid">
          <!-- Left: Original Video with Bounding Box -->
          <div class="video-panel">
            <div class="video-panel-header">
              <span class="video-panel-title">Raw Video (Bounding Box)</span>
            </div>
            <video id="vid-left" playsinline muted preload="auto">
              <source src="data:video/mp4;base64,{v_left_b64}" type="video/mp4">
            </video>
          </div>

          <!-- Right: Model Output (Prediction Error Heatmap) -->
          <div class="video-panel">
            <div class="video-panel-header">
              <span class="video-panel-title">Prediction Error Heatmap</span>
            </div>
            <video id="vid-right" playsinline muted preload="auto">
              <source src="data:video/mp4;base64,{v_right_b64}" type="video/mp4">
            </video>
          </div>
        </div>

        <!-- Unified Synchronized Controller Bar -->
        <div class="sync-controller">
          <button id="btn-play" class="btn-play" title="Play / Pause">▶</button>
          <button id="btn-prev" class="btn-step" title="Step -1 frame">⏮</button>
          <button id="btn-next" class="btn-step" title="Step +1 frame">⏭</button>
          <span id="time-text" class="time-display">00:00 / 00:00</span>
          
          <div class="scrubber-wrap">
            <input type="range" id="seek-bar" class="scrubber" min="0" max="100" value="0" step="0.03">
          </div>
          
          <select id="speed-select" class="speed-select" title="Playback Speed">
            <option value="0.5">0.5x</option>
            <option value="1.0" selected>1.0x</option>
            <option value="1.5">1.5x</option>
            <option value="2.0">2.0x</option>
          </select>
        </div>
      </div>

      <script>
        const v1 = document.getElementById('vid-left');
        const v2 = document.getElementById('vid-right');
        const btnPlay = document.getElementById('btn-play');
        const seekBar = document.getElementById('seek-bar');
        const timeText = document.getElementById('time-text');
        const speedSelect = document.getElementById('speed-select');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');

        let isSeeking = false;
        const fps = {fps if fps > 0 else 30.0};
        const frameTime = 1.0 / fps;

        function formatTime(s) {{
          if (isNaN(s) || s < 0) return '00:00';
          const mins = Math.floor(s / 60);
          const secs = Math.floor(s % 60);
          return (mins < 10 ? '0' : '') + mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}

        function updateDuration() {{
          const dur = Math.max(v1.duration || 0, v2.duration || 0);
          if (dur > 0) {{
            seekBar.max = dur;
            timeText.textContent = formatTime(v1.currentTime) + ' / ' + formatTime(dur);
          }}
        }}

        v1.addEventListener('loadedmetadata', updateDuration);
        v2.addEventListener('loadedmetadata', updateDuration);

        // Unified Play / Pause
        function togglePlay() {{
          if (v1.paused || v2.paused) {{
            v2.currentTime = v1.currentTime;
            v1.play();
            v2.play();
            btnPlay.innerHTML = '⏸';
          }} else {{
            v1.pause();
            v2.pause();
            btnPlay.innerHTML = '▶';
          }}
        }}
        btnPlay.addEventListener('click', togglePlay);
        v1.addEventListener('click', togglePlay);
        v2.addEventListener('click', togglePlay);

        // Sync & Drift Lock
        v1.addEventListener('timeupdate', () => {{
          if (!isSeeking) {{
            seekBar.value = v1.currentTime;
            const dur = Math.max(v1.duration || 0, v2.duration || 0);
            timeText.textContent = formatTime(v1.currentTime) + ' / ' + formatTime(dur);

            // Auto-align if drift exceeds 0.05s
            if (Math.abs(v1.currentTime - v2.currentTime) > 0.05) {{
              v2.currentTime = v1.currentTime;
            }}
          }}
        }});

        // Scrubber Seeking
        seekBar.addEventListener('input', () => {{
          isSeeking = true;
          const t = parseFloat(seekBar.value);
          v1.currentTime = t;
          v2.currentTime = t;
          const dur = Math.max(v1.duration || 0, v2.duration || 0);
          timeText.textContent = formatTime(t) + ' / ' + formatTime(dur);
        }});

        seekBar.addEventListener('change', () => {{
          isSeeking = false;
          const t = parseFloat(seekBar.value);
          v1.currentTime = t;
          v2.currentTime = t;
        }});

        // Frame Stepping
        btnPrev.addEventListener('click', () => {{
          v1.pause(); v2.pause(); btnPlay.innerHTML = '▶';
          const t = Math.max(0, v1.currentTime - frameTime);
          v1.currentTime = t; v2.currentTime = t;
          seekBar.value = t;
        }});

        btnNext.addEventListener('click', () => {{
          v1.pause(); v2.pause(); btnPlay.innerHTML = '▶';
          const dur = Math.max(v1.duration || 0, v2.duration || 0);
          const t = Math.min(dur, v1.currentTime + frameTime);
          v1.currentTime = t; v2.currentTime = t;
          seekBar.value = t;
        }});

        // Speed Adjustment
        speedSelect.addEventListener('change', () => {{
          const spd = parseFloat(speedSelect.value);
          v1.playbackRate = spd;
          v2.playbackRate = spd;
        }});

        // Video End Sync
        v1.addEventListener('ended', () => {{
          v2.pause();
          btnPlay.innerHTML = '▶';
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=450)


# Alias for backward compatibility
render_synchronized_triple_video = render_synchronized_dual_video



# ==============================================================================
# 4. Main Page Rendering
# ==============================================================================
def main():
    # Page Configuration
    st.set_page_config(
        page_title="Anomaly Detection Studio",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Apply CSS
    apply_custom_styles()

    # Session State
    if "results" not in st.session_state:
        st.session_state.results = None
    if "uploaded_video_path" not in st.session_state:
        st.session_state.uploaded_video_path = None
    if "uploaded_video_name" not in st.session_state:
        st.session_state.uploaded_video_name = None

    import torch
    device_tag = "CUDA:0 (GPU)" if torch.cuda.is_available() else "CPU"

    # Centered Title & Architecture Badge Bar
    st.markdown(
        f"""
        <div class="app-header-center">
            <h1 class="app-header-title">Anomaly detection</h1>
            <div class="model-badge-bar">
                <span class="model-badge-item">⚙️ Config: <b>ped2_wresnet.yaml</b></span>
                <span class="model-badge-item">🧠 Model: <b>ASTNet (WideResNet-38)</b></span>
                <span class="model-badge-item">🎯 Dataset: <b>UCSD Ped2</b></span>
                <span class="model-badge-item">📦 Weights: <b>checkpoints/ped2.pth</b></span>
                <span class="model-badge-item">⚡ Device: <b>{device_tag}</b></span>
                <span class="model-badge-item">📐 Input: <b>224 × 288 px (T=4)</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Unified Card: Video Input & Configuration
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-top-header">
                <div class="card-main-title">📥 Video Input & Detection Configuration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            label="Upload Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV",
            label_visibility="collapsed",
        )

        display_video_name = None
        display_video_size = None
        if uploaded_file is not None:
            display_video_name = uploaded_file.name
            display_video_size = f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        elif st.session_state.uploaded_video_name is not None and st.session_state.uploaded_video_path and os.path.exists(st.session_state.uploaded_video_path):
            display_video_name = st.session_state.uploaded_video_name
            display_video_size = f"{os.path.getsize(st.session_state.uploaded_video_path) / (1024 * 1024):.2f} MB"

        if display_video_name:
            st.markdown(
                f"""
                <div class="file-chip">
                    <span>🎥 <b>File:</b> {display_video_name}</span>
                    <span>•</span>
                    <span>{display_video_size}</span>
                    <span>•</span>
                    <span style="color: #16A34A;">✓ Ready for Analysis</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)

        # Parameter Controls row inside the Unified Card
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2.4, 1.0, 1.0], vertical_alignment="bottom")

        with col_ctrl1:
            threshold = st.slider(
                "🎯 Anomaly Detection Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Frames with scores above this threshold are classified as anomalies.",
            )

        with col_ctrl2:
            analyze_clicked = st.button(
                "Phân tích",
                icon=":material/play_arrow:",
                type="primary",
                use_container_width=True,
            )

        with col_ctrl3:
            reset_clicked = st.button(
                "Reset",
                icon=":material/refresh:",
                type="secondary",
                use_container_width=True,
            )

    # Reset Action
    if reset_clicked:
        st.session_state.results = None
        st.session_state.uploaded_video_path = None
        st.session_state.uploaded_video_name = None
        st.rerun()

    # Cache uploaded file locally
    if uploaded_file is not None:
        if st.session_state.uploaded_video_name != uploaded_file.name:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_video_path = temp_path
            st.session_state.uploaded_video_name = uploaded_file.name
            st.session_state.results = None

    # Analysis Execution
    if analyze_clicked:
        if st.session_state.uploaded_video_path is None:
            st.warning("⚠️ Please drop a video into the upload zone first.")
        else:
            status_text = st.empty()
            status_text.write("Đang khởi tạo model")

            def update_ui_progress(current, total=None):
                if total is not None:
                    status_text.write(f"{current}/{total}")
                else:
                    status_text.write(str(current))

            try:
                res = run_anomaly_detection(
                    st.session_state.uploaded_video_path,
                    threshold=threshold,
                    progress_callback=update_ui_progress,
                )
                status_text.empty()
                st.session_state.results = res
                st.rerun()
            except Exception as e:
                status_text.empty()
                st.error(f"Đã xảy ra lỗi trong quá trình suy luận: {str(e)}")

    # Results Dashboard
    if st.session_state.results is not None:
        res = st.session_state.results
        is_anomaly = res["peak_score"] >= threshold

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # KPI Summary Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        status_title = "ANOMALY DETECTED" if is_anomaly else "NORMAL FEED"
        status_color = "#DC2626" if is_anomaly else "#16A34A"
        status_bg = "#FEF2F2" if is_anomaly else "#F0FDF4"
        status_border = "#FECACA" if is_anomaly else "#BBF7D0"

        with col_m1:
            st.markdown(
                f"""
                <div class="metric-box" style="background: {status_bg}; border-color: {status_border};">
                    <div class="metric-header" style="color: {status_color};">Detection Status</div>
                    <div class="metric-number" style="color: {status_color}; font-size: 1.3rem;">{status_title}</div>
                    <div class="metric-subtext" style="color: {status_color};">Threshold: {threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            peak_color = "#DC2626" if is_anomaly else "#2563EB"
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-header">Peak Anomaly Score</div>
                    <div class="metric-number" style="color: {peak_color};">{res['peak_score']:.3f}</div>
                    <div class="metric-subtext">Max sequence score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-header">Video Duration</div>
                    <div class="metric-number">{res['duration']:.1f}s</div>
                    <div class="metric-subtext">{res['total_frames']} frames @ {res['fps']:.1f} FPS</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-header">Inference Time</div>
                    <div class="metric-number">{res['inference_time']:.2f}s</div>
                    <div class="metric-subtext">Spatiotemporal inference</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # Synchronized Dual Video Display Screens (Raw Video | Error Heatmap)
        annotated_video_path = res.get("annotated_video_path", res.get("processed_video_path"))
        heatmap_video_path = res.get("heatmap_video_path")

        base_name = ""
        if st.session_state.uploaded_video_path:
            base_name = os.path.splitext(os.path.basename(st.session_state.uploaded_video_path))[0]
        if not base_name and annotated_video_path:
            raw_base = os.path.splitext(os.path.basename(annotated_video_path))[0]
            base_name = raw_base.replace("_annotated_web", "").replace("_annotated", "")

        # If not present in cached results, check if pre-rendered heatmap video exists in temp
        if not heatmap_video_path or not os.path.exists(heatmap_video_path):
            if base_name:
                temp_candidate = os.path.join(tempfile.gettempdir(), f"{base_name}_error_heatmap.mp4")
                if os.path.exists(temp_candidate):
                    heatmap_video_path = temp_candidate
                    res["heatmap_video_path"] = temp_candidate

        right_video_path = (
            heatmap_video_path
            if (heatmap_video_path and os.path.exists(heatmap_video_path))
            else st.session_state.uploaded_video_path
        )

        if os.path.exists(annotated_video_path) and os.path.exists(right_video_path):
            render_synchronized_dual_video(
                video_left_path=annotated_video_path,
                video_right_path=right_video_path,
                fps=res["fps"],
            )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # Interactive Anomaly Score Chart (Plotly)
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-top-header" style="margin-bottom: 0.5rem;">
                    <div class="card-main-title">Temporal Anomaly Score Line Chart</div>                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            # Blue area curve for anomaly score
            fig.add_trace(
                go.Scatter(
                    x=res["timestamps"],
                    y=res["scores"],
                    mode="lines",
                    name="Anomaly Score",
                    line=dict(color="#2563EB", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.08)",
                    hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Score:</b> %{y:.3f}<extra></extra>",
                )
            )

            # Red dashed threshold line
            fig.add_trace(
                go.Scatter(
                    x=[res["timestamps"][0], res["timestamps"][-1]],
                    y=[threshold, threshold],
                    mode="lines",
                    name=f"Threshold ({threshold:.2f})",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                    hovertemplate=f"Threshold: {threshold:.2f}<extra></extra>",
                )
            )

            # Highlight points above threshold
            anomaly_times = [t for t, s in zip(res["timestamps"], res["scores"]) if s >= threshold]
            anomaly_vals = [s for s in res["scores"] if s >= threshold]
            if anomaly_times:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_times,
                        y=anomaly_vals,
                        mode="markers",
                        name="Anomaly Flagged",
                        marker=dict(color="#DC2626", size=5, symbol="circle"),
                        hovertemplate="<b>Anomaly Detected!</b><br>Time: %{x:.2f}s<br>Score: %{y:.3f}<extra></extra>",
                    )
                )

            # Modern Plotly theme
            fig.update_layout(
                margin=dict(l=40, r=40, t=30, b=40),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12, color="#475569"),
                ),
                xaxis=dict(
                    title=dict(text="Time (seconds)", font=dict(size=12, color="#64748B")),
                    showgrid=True,
                    gridcolor="#F1F5F9",
                    zeroline=False,
                    showline=True,
                    linecolor="#CBD5E1",
                    tickfont=dict(color="#64748B"),
                ),
                yaxis=dict(
                    title=dict(text="Anomaly Score", font=dict(size=12, color="#64748B")),
                    range=[0, 1.05],
                    showgrid=True,
                    gridcolor="#F1F5F9",
                    zeroline=True,
                    zerolinecolor="#E2E8F0",
                    showline=True,
                    linecolor="#CBD5E1",
                    tickfont=dict(color="#64748B"),
                ),
                height=360,
            )

            st.plotly_chart(fig, use_container_width=True)  

if __name__ == "__main__":
    main()
