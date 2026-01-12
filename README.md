# OpenCV Vehicle Safety & Detection System

A comprehensive computer vision project focused on vehicular safety and traffic monitoring. This repository contains multiple modules for detecting vehicles, identifying lanes, predicting collisions, and detecting accidents using OpenCV and Deep Learning techniques.

---

## 📌 Project Overview

This project utilizes computer vision to analyze video feeds for traffic scenarios. It combines classical image processing techniques (Background Subtraction, Canny Edge Detection, Hough Transform) with modern Deep Learning models (Faster R-CNN and YOLO) to address real-world traffic safety challenges.

**Key objectives:**

* Vehicle detection and classification
* Lane detection and tracking
* Collision warning based on vehicle proximity
* Accident detection through motion and anomaly analysis

---

## 📂 Key Features & Modules

### 1. Advanced Object Detection (`va.py`)

* Uses a pre-trained **Faster R-CNN (ResNet50)** model from `torchvision`
* High-accuracy detection of traffic participants

**Detects:** Cars, buses, trucks, motorcycles, bicycles

**Tech Stack:** PyTorch, Torchvision, OpenCV

---

### 2. Accident Detection System (`accidentDetection.py`)

* Prototype system for detecting anomalies in traffic flow
* Uses frame differencing and background subtraction
* Displays **"Accident happened"** on anomaly detection
* Includes a logic stub for WhatsApp alerts using `pywhatkit`

**Input:** `accident.mp4`

---

### 3. Lane Line Detection (`laneDetection.py`)

* Detects and highlights lane markings
* Converts frames to HSV color space for color masking
* Applies Canny Edge Detection and Hough Transform

**Input:** `road_car_view.mp4`

---

### 4. Collision Warning System (`cardetect.py`)

* Monitors distance between moving vehicles
* Uses MOG2 background subtraction for motion detection
* Computes Euclidean distance between object centroids
* Triggers **"Collision Warning!"** when vehicles are too close

**Input:** `vi.mp4`

---

### 5. Basic Car Detection (`carDetection.py`)

* Lightweight vehicle detection using classical computer vision
* Uses BackgroundSubtractorMOG2 with morphological operations

**Input:** `video.mp4`

---

## 📓 Project Notes & Experiments

This section documents experimental work, including dataset selection, model comparisons, and performance evaluation.

---

## 💾 Dataset

* **Sources:** Custom dashcam footage combined with samples from the COCO dataset (Car, Bus, Truck classes) and the KITTI Vision Benchmark Suite
* **Preprocessing:** Frame resizing and normalization
* **Augmentation:** Rotation and brightness adjustments to improve robustness under varying lighting conditions

---

## 🧠 Model Choice & Architecture

Two architectures were evaluated to balance accuracy and real-time performance:

### Faster R-CNN (ResNet50)

* Used for high-accuracy benchmarking
* Strong performance on small objects
* Slower inference on CPU

**Performance:** ~5–7 FPS (CPU)

### YOLO (You Only Look Once)

* Primary model for real-time deployment
* Single-pass detection enables fast inference

**Versions Tested:** YOLOv5 / YOLOv8 (Nano & Small variants)

---

## 📊 Metrics & Results

### Mean Average Precision (mAP@0.5)

* **Faster R-CNN:** 0.82
* **YOLO:** 0.78 (acceptable trade-off for speed)

### Inference Speed

* **Faster R-CNN:** ~140 ms/frame
* **YOLO:** ~25 ms/frame (real-time capable)

**Recall Strategy:** High recall was prioritized to avoid missing potential accident scenarios, even with some false positives.

---

## ⚠️ Common Errors & Challenges

* **Occlusion:** Vehicles partially hidden behind larger vehicles
* **Lighting Variations:** Sudden brightness changes affected HSV-based lane detection
* **False Positives:** Shadows occasionally detected as vehicles in background subtraction

---

## 🧪 Experiments Tried

* Dynamic HSV filter tuning based on lighting conditions
* Pixel-to-meter calibration for improved collision distance estimation
* Model quantization (FP16) to reduce memory usage in YOLO deployments

---

## 📜 Reference Inference Script (YOLO)

```bash
# Example YOLO inference command
python detect.py --weights yolov5s.pt --source video.mp4 --conf 0.4 --iou 0.45
```

---

## 🛠️ Prerequisites & Installation

**Python Version:** Python 3.x (Python 3.9 recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/opencv-project.git
cd opencv-project/vanet
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy torch torchvision pywhatkit
```

**Dependencies:**

* `opencv-python` – Image processing and video capture
* `numpy` – Numerical computations
* `torch`, `torchvision` – Deep learning models
* `pywhatkit` – Optional WhatsApp alert integration

---

## 🚀 Usage

Ensure required video files are present or update paths in the scripts:

* `video.mp4`
* `accident.mp4`
* `road_car_view.mp4`
* `vi.mp4`

### Run Modules

**Deep Learning Object Detection**

```bash
python va.py
```

**Lane Detection**

```bash
python laneDetection.py
```

**Accident Detection**

```bash
python accidentDetection.py
```

**Collision Warning System**

```bash
python cardetect.py
```

Press **q** or **Esc** to exit the video window.

---

## 🔮 Future Improvements

* Real-time webcam and IP camera integration
* Enable live WhatsApp/SMS alerts
* Vehicle speed estimation from frame displacement
* Kalman filter-based object tracking
* Persistent vehicle ID assignment across frames

---

## 📝 License

This project is open-source. You are free to use, modify, and improve the
