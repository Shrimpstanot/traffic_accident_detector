# Real-Time Traffic Accident Detection using a Hierarchical Deep Learning Model

This repository contains the code for a high-performance, real-time traffic accident detection system. The project's core innovation is a **two-level hierarchical architecture** that mirrors human reasoning: it first understands the visual content of short video clips and then analyzes their sequence to understand the overall narrative of a traffic scene.

This approach was specifically engineered to solve the "context is everything" problem, where pre-crash and normal traffic clips are visually indistinguishable. By analyzing the temporal story, the system achieves high-precision results on the challenging task of real-time event detection.

The entire pipeline is built to be deployable, featuring a multi-threaded, "chunk-based" inference architecture that maximizes GPU utilization and ensures stable, real-time performance on headless servers.

---

### 🎥 Live Demo

![Real-Time Crash Detection Demo](https://i.imgur.com/5krQLsx.gif) 


---

### ✨ Key Features

| Feature                  | Description                                                                                                                                                             | Status      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| **Hierarchical AI Model**| A sophisticated CNN+RNN architecture. An **X3D** model extracts features from clips, and an **LSTM** analyzes the feature sequence to detect the temporal pattern of a crash. | ✅ Complete |
| **High-Performance Pipeline** | A multi-threaded, chunk-based inference engine designed for real-time, headless deployment. Maximizes performance by decoupling I/O and GPU workloads.                     | ✅ Complete |
| **Two Runtime Modes**      | Run in **headless mode** to generate an annotated `.mp4` file (perfect for servers/Docker) or in **display mode** to see a live annotated feed from a video or webcam. | ✅ Complete |
| **Optional Object Tracking** | Includes a pre-trained **YOLOv8** model to provide rich vehicle and pedestrian tracking, which can be run alongside the crash detection system for comprehensive scene understanding. | ✅ Optional |

---

### 📂 Repository Structure

```
.
├── checkpoints/                 # Pre-trained X3D classifier, Detector, and LSTM
├── demo/                        # Contains 2 processed videos
├── realtime_headless.py         # Headless version or realtime_pipeline.py
├── realtime_pipeline.py         # The main real-time inference script (headless & display modes)
├── inference.py                 # Offline script for analyzing a single video file
├── train_classifier.py          # Code for training the X3D model
├── train_lstm.py                # Code for training the LSTM model
├── requirements.txt             # Python dependencies
└── README.md
```

---

### ⚙️ Getting Started

#### 1. Clone the Repository and Install Dependencies

```bash
git clone https://github.com/yourusername/traffic-accident-detection.git
cd traffic-accident-detection
pip install -r requirements.txt
```

#### 2. Run the Pipeline

The `realtime_pipeline.py` script is the main entry point. You can control its behavior using arguments.

**A) To process a video file in headless mode (Recommended for first run):**

This will analyze the specified input video and save an annotated version to `output.mp4`.

```bash
python realtime_headless.py --input_video path/to/your/video.mp4 --output_video output.mp4
```

**B) To run in display mode with a video file:**

This will open an OpenCV window and show the live annotated feed. Press `q` to quit.

```bash
python realtime_pipeline.py --input_video path/to/your/video.mp4
```

**C) To run in display mode with a webcam:**

```bash
python realtime_pipeline.py --use_webcam
```

---

### 🧠 The Models & Training

This project's success relies on a multi-stage training process that addresses key real-world AI challenges.

1.  **The Core Insight: A Crash is a Narrative, Not an Image.**
    -   My initial analysis using t-SNE visualization proved that pre-crash and normal traffic clips are visually identical. This insight drove the decision to move from a simple classifier to a hierarchical model that could understand temporal context.

2.  **Level 1: The X3D "Clip Reader"**
    -   **Dataset:** [TU-DAT (Traffic Accident Dataset)](<https://www.mdpi.com/1424-8220/25/11/3259>)
    -   **Architecture:** An X3D-M model, pre-trained on Kinetics-400, was fine-tuned on TU-DAT clips to become an expert feature extractor for short traffic-related video segments.

3.  **Level 2: The LSTM "Storyteller"**
    -   **Dataset:** Sequences of feature vectors extracted by the X3D model.
    -   **Architecture:** An LSTM network was trained on these sequences to learn the temporal patterns that distinguish a crash event from normal traffic flow.

4.  **Optional: The YOLOv8 "Observer"**
    -   **Datasets:** A combination of [UA-DETRAC](<https://universe.roboflow.com/rjacaac1/ua-detrac-dataset-10k/dataset/2>) (for vehicles) and [MOTChallenge](<https://motchallenge.net/>) (for pedestrians).
    -   **Purpose:** Provides rich object-level metadata (bounding boxes, track IDs) for comprehensive scene awareness. Can be added for car tracking during inference.

---

### 📑 References & Acknowledgments

-   This work is built upon the foundational research of Feichtenhofer et al., [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/abs/2004.04730) (CVPR 2020).
-   This project would not have been possible without the public datasets provided by the **TU-DAT**, **UA-DETRAC**, and **MOTChallenge** teams.
-   Built with the incredible tools from the **PyTorch**, **PyTorch Lightning**, and **OpenCV** communities.
