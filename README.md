# 🛡️ Video Anomaly Detection Studio

> **Real-time Video Anomaly Scoring & Temporal Localization**  
> A modern, interactive web application built with **Streamlit**, **OpenCV**, and **Plotly** for analyzing surveillance video footage and detecting anomalous events.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
  - [Method 1: Using Makefile (Recommended)](#method-1-using-makefile-recommended)
  - [Method 2: Using Python & Virtual Environment](#method-2-using-python--virtual-environment)
- [Makefile Commands](#-makefile-commands)
- [UI Usage Guide](#-ui-usage-guide)
- [Model Integration & Architecture](#-model-integration--architecture)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## ✨ Features

- 📹 **Flexible Video Upload**: Drag & drop or browse video files supporting `.mp4`, `.avi`, `.mov`, and `.mkv`.
- 🎯 **Adjustable Sensitivity Threshold**: Interactive slider (0.10 to 0.90) to fine-tune anomaly decision boundaries in real time.
- ⚡ **Real-Time Temporal Inference**: Extracts temporal motion features and computes frame-by-frame anomaly confidence scores.
- 📊 **Executive KPI Metrics**:
  - **Detection Status**: Instant classification badge (`NORMAL FEED` vs. `ANOMALY DETECTED`).
  - **Peak Anomaly Score**: Maximum anomaly score detected in the video stream.
  - **Video Duration & FPS**: Stream metadata including duration and frame count.
  - **Inference Latency**: Precise timing for inference and feature extraction.
- 🔄 **Synchronized Dual Video Player**:
  - Side-by-side synchronized comparison (Input video vs. Processed/Annotated stream).
  - Unified controls: Play / Pause, Seek bar, Previous/Next frame-stepping, and Playback speed selector (`0.5x`, `1.0x`, `1.5x`, `2.0x`).
- 📈 **Interactive Temporal Anomaly Chart**:
  - Interactive Plotly area chart visualizing confidence scores over time.
  - Visual threshold line indicator.
  - Flagged anomaly markers highlighting exact violation timestamps with hover details.

---

## 📁 Project Structure

```text
Anomaly-Detection/
├── anomaly-detection/
│   ├── app.py                      # Main entrypoint to launch Streamlit UI
│   ├── Makefile                    # Make command shortcuts (run, install, clean, etc.)
│   ├── requirements.txt            # Python package dependencies
│   ├── website_interface/
│   │   ├── __init__.py
│   │   ├── ui.py                   # UI layout, CSS styles, dual player & inference hook
│   │   └── .streamlit/
│   │       └── config.toml         # Streamlit UI theme configuration
│   ├── checkpoints/
│   │   └── wider_resnet38.pth      # Pretrained model weights
│   ├── configs/                    # Model and pipeline configuration files
│   ├── data/                       # Sample test videos and datasets
│   ├── models/                     # Deep learning model architectures
│   └── src/                        # Core utilities, metrics, and video processor
├── Makefile                        # Root Makefile forwarding commands to project
└── README.md                       # Project documentation
```

---

## ⚙️ Requirements

- **Operating System**: Linux / macOS / Windows
- **Python**: Version `3.10` or newer (tested with Python `3.14`)
- **Key Python Packages**:
  - `streamlit >= 1.30.0`
  - `plotly >= 5.18.0`
  - `opencv-python >= 4.8.0`
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `torch >= 2.0.0`
  - `torchvision >= 0.15.0`

---

## 🚀 Quick Start

### Method 1: Using Make (Simplest)

The project includes a streamlined `Makefile` configured specifically to launch the UI:

```bash
make ui
```
*or simply run:*
```bash
make
```
*The application will launch and open at `http://localhost:8501`.*

To run on a custom port or host:
```bash
make ui PORT=8080 HOST=127.0.0.1
```

---

### Method 2: Using Python Directly

1. **Activate your environment** (e.g. `source .venv/bin/activate`)
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Makefile Usage

The `Makefile` is dedicated to running the Streamlit UI:

```bash
make ui           # Starts UI on http://localhost:8501
make ui PORT=8080 # Starts UI on custom port 8080
```

> [!TIP]
> You can run `make ui` from either the root directory or inside the `anomaly-detection/` subdirectory.

---

## 🖥️ UI Usage Guide

### 1. Upload Video
- Click on the upload card or drag-and-drop a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
- The application validates the file and displays its name, size, and readiness status.

### 2. Configure Sensitivity Threshold
- Use the **Anomaly Detection Threshold** slider (default: `0.50`).
- Lower values (e.g. `0.30`) make the detector more sensitive to minor changes or unexpected movements.
- Higher values (e.g. `0.70`) restrict anomaly alerts to only severe, high-confidence events.

### 3. Run Analysis
- Click the primary blue button **`▶ Analyze Video`**.
- The real-time progress banner displays the processing pipeline: decoding streams, extracting spatiotemporal features, and computing anomaly scores.

### 4. Review Results & Diagnostics
- **Status Badge**: Clear visual banner indicating `NORMAL FEED` (green) or `ANOMALY DETECTED` (red).
- **Synchronized Dual Video Player**:
  - Left panel: Input feed.
  - Right panel: Processed/annotated stream.
  - Unified playback: Clicking Play/Pause or dragging the progress slider stays in sync across both streams.
  - Frame-stepping: Use `< Frame` and `Frame >` buttons for forensic inspection.
- **Plotly Temporal Chart**:
  - Hover over the curve to inspect timestamp and exact anomaly score.
  - Detected anomalies are clearly highlighted above the red dashed threshold line.

### 5. Reset Studio
- Click **`🔄 Reset Studio`** to clear previous results and upload a new video.

---

## 🧠 Model Integration & Architecture

The user interface delegates inference through the hook function located in `website_interface/ui.py`:

```python
def run_anomaly_detection(video_path: str, threshold: float = 0.5):
    """
    Hook your deep learning anomaly detection model here.
    
    Args:
        video_path: Path to the uploaded video file.
        threshold: Anomaly score decision threshold (0.0 to 1.0).
        
    Returns:
        dict containing processed_video_path, timestamps, scores, peak_score, etc.
    """
```

To plug in a custom deep learning model (e.g., using `checkpoints/wider_resnet38.pth` or a 3D CNN / Transformer model):
1. Load your model weights inside or before `run_anomaly_detection()`.
2. Extract spatiotemporal features from frames extracted using `cv2.VideoCapture`.
3. Normalize output anomaly scores between `0.0` and `1.0`.
4. (Optional) Annotate bounding boxes or heatmaps onto the video and save the path to `processed_video_path`.

---

## ❓ Troubleshooting & FAQ

### 1. `streamlit: command not found`
If `streamlit` was installed in your user directory (e.g., `~/.local/bin`), ensure it is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Or run directly via python module:
```bash
python3 -m streamlit run app.py
```

### 2. Video codec issues in browser player
Browsers natively support H.264 encoded MP4 files. If an uploaded `.avi` or `.mkv` video does not render in the HTML5 player, convert it to H.264 MP4:
```bash
ffmpeg -i input_video.avi -vcodec libx264 -acodec aac output.mp4
```

### 3. Port already in use
If port `8501` is already in use by another process:
```bash
make run PORT=8502
```
