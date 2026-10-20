<div align="center">

# 🛡️ IronShield: Tactical Air Defense & Object Interception ML System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLO-v5%20%7C%20v8-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%20%2F%20CPU-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Status](https://img.shields.io/badge/Deployment-Operational-brightgreen.svg)]()

**A state-of-the-art AI-powered tactical air defense, radar tracking, and missile interception simulation engine built with PyTorch, YOLO Neural Networks, and OpenCV.**

Developed by **[Vilochana Rajapaksha](https://github.com/sadisi)**

---

</div>

## 📌 Overview

**IronShield** is an advanced Computer Vision and Deep Learning air-defense monitoring system designed to detect, track, lock onto, and calculate interception vectors for aerial threats (aircraft, drones, missiles, unidentified aerial objects) in real time.

Equipped with a sleek military-grade tactical Heads-Up Display (HUD), dynamic targeting reticles, real-time telemetry panels, and audible warning sirens, IronShield simulates modern multi-layered air defense command consoles.

```
       [ Live Optical / Radar Feed ]
                     │
                     ▼
       [ YOLOv5 / YOLOv8 Detection Net ]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ Aerial Threat Filter ]   [ Vehicle / Ground Target ]
       │                           │
       ▼                           ▼
[ Target Lock & Telemetry ] [ Secondary Tracking ]
       │
       ├─► [ Audio Siren Alarm (Pygame) ]
       ├─► [ Interception Vector Calculation ]
       └─► [ Anti-Aliased Tactical Military HUD ]
```

---

## ✨ Key Features

- 🎯 **Deep Learning Threat Recognition**: Uses pre-trained and fine-tunable YOLOv5 and YOLOv8 models optimized for aerial and vehicular target classification.
- 📐 **Precision Target Lock & Tracking**: Automatically identifies the highest-threat priority target, calculating dynamic center-mass coordinates (Azimuth X/Y).
- 🚀 **Missile Interception Simulation**: Computes dynamic defense vectors and simulated interception trajectories from ground battery positions to incoming hostiles.
- 🖥️ **Tactical Military HUD**:
  - High-visibility corner-bracket target markers.
  - Active crosshairs and dynamic targeting rings.
  - Telemetry console displaying Target Classification, Confidence %, Coordinates, and Lock Status.
  - Live FPS and Hardware Engine monitoring (NVIDIA GPU / CUDA or adaptive CPU).
- 🚨 **Acoustic Early-Warning Alarm**: Automatically triggers a tactical audible siren (`warn.wav`) via Pygame whenever aerial threats breach perimeter airspace.
- 🎮 **Tactical Operator Controls**: Live hotkeys for pause/resume, feed switching, muting siren audio, and one-click forensic snapshot capture.
- ⚡ **Multi-Hardware Support**: Auto-detects NVIDIA CUDA GPUs (e.g., GTX 1650, RTX series) for high FPS inference with graceful fallback to CPU.

---

## 🗂️ Project Structure

```bash
Air-Defense-System-ML/
│
├── ironshield.py          # 🛡️ Primary Tactical Defense Console (HUD, Interception, Audio Alarm)
├── lock.py                # 🎯 Precision Radar Target Lock & Azimuth Telemetry Console
├── ditect.py              # 👁️ High-Speed Optical Detection & Classification Engine
├── imp.py                 # 🧪 Core YOLO Detection & Labeling Utility
├── yolov5s.pt             # 🧠 Pre-trained YOLOv5 Neural Network Weights
├── requirements.txt       # 📦 Python Dependencies
├── warn.wav               # 🔊 Tactical Warning Siren Sound
│
├── imp/                   # 🔬 Experimental YOLOv8 Engine
│   ├── Tracker.py         # YOLOv8 tracking module
│   ├── Ditect.py          # YOLOv8 detection module
│   └── yolov8n.pt         # YOLOv8 nano model weights
│
└── vid/                   # 🎥 Tactical Sample Videos (Aerial & Combat Feeds)
    ├── 15285393_1080_1922_60fps.mp4
    ├── 3721818-hd_1920_1080_30fps.mp4
    ├── 15172694-uhd_2160_3840_60fps.mp4
    ├── 14792009_2160_3840_30fps.mp4
    ├── 1585496-hd_1920_1080_30fps.mp4
    └── 15076427_3840_2160_30fps.mp4
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/sadisi/Air-Defense-System-ML.git
cd Air-Defense-System-ML
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch CUDA Support**: For GPU acceleration with NVIDIA graphics cards, install the PyTorch build corresponding to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🎮 Running the Tactical Consoles

### Option 1: Full IronShield Air Defense System

Launch the complete air defense command system featuring tactical interception vector lines, dynamic telemetry, threat monitoring, and siren alerts:

```bash
python ironshield.py
```

### Option 2: Precision Target Lock Console

Launch the dedicated target acquisition and precision tracking radar HUD:

```bash
python lock.py
```

### Option 3: Optical Reconnaissance Console

Launch the lightweight optical target classification engine:

```bash
python ditect.py
```

---

## ⌨️ Operator Keybindings & Controls

While any tactical console window is active:

| Key | Action | Description |
| :---: | :--- | :--- |
| **`[SPACE]`** | **Pause / Resume** | Freeze frame for tactical inspection |
| **`[M]`** | **Mute / Unmute** | Toggle the audible early-warning siren (`warn.wav`) |
| **`[N]`** | **Next Video Feed** | Cycle to the next video in the `vid/` playlist |
| **`[S]`** | **Forensic Snapshot** | Save timestamped full-resolution frame capture |
| **`[Q]` / `[ESC]`** | **Safe Exit** | Safely terminate the console and release resources |

---

## 📊 Tactical HUD & Telemetry Specification

The IronShield HUD renders real-time telemetry overlays directly onto optical streams:

- **Top Status Banner**: Displays system operational state (`DEFENSE GRID ACTIVE`), threat counter (`THREATS: N` / `AIRSPACE: SECURE`), and hardware compute engine (`GPU / CPU` + real-time FPS).
- **Target Reticles**: Anti-aliased corner brackets around acquired targets with confidence ratings.
- **Precision Crosshairs**: Center-mass tracking lines spanning the target azimuth and elevation.
- **Threat Telemetry Box**: Displays:
  - Target Classification (`AIRPLANE`, `MISSILE`, `DRONE`, etc.)
  - Neural Network Probability Percentage
  - Azimuth Grid Coordinates `(X, Y)`
  - Defense Status (`ACTIVE TARGET LOCK`)
- **Bottom Status Bar**: Interactive keybindings reminder and active feed source filename.

---

## 🛠️ Tech Stack & Dependencies

- **Computer Vision**: [OpenCV (opencv-python)](https://opencv.org/)
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) & [TorchVision](https://pytorch.org/vision/)
- **Object Detection Models**: [Ultralytics YOLOv5 & YOLOv8](https://github.com/ultralytics)
- **Audio Synthesis**: [Pygame / Pygame-CE](https://pyga.me/)
- **Numerical Processing**: [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)

---

## 👤 Author

**Vilochana Rajapaksha**  
- GitHub: [@sadisi](https://github.com/sadisi)
- Repository: [Air-Defense-System-ML](https://github.com/sadisi/Air-Defense-System-ML)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
