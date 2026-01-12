# OpenCV Vehicle Safety & Detection System

A comprehensive computer vision project focused on vehicular safety and traffic monitoring. This repository contains multiple modules for detecting vehicles, identifying lanes, predicting collisions, and detecting accidents using OpenCV and Deep Learning techniques.

---

## 📌 Project Overview

This project analyzes traffic video feeds using computer vision techniques. It combines classical image processing (Background Subtraction, Canny Edge Detection, Hough Transform) with Deep Learning (Faster R-CNN) to address real-world traffic safety problems.

**Core functionalities include:**

* Vehicle detection and classification
* Lane detection and tracking
* Collision warning based on vehicle proximity
* Accident detection using motion analysis

---

## 📂 Key Features & Modules

### 1. Advanced Object Detection (`va.py`)

* Uses a pre-trained **Faster R-CNN (ResNet50)** model from `torchvision`
* Detects: cars, buses, trucks, motorcycles, bicycles
* High-accuracy object detection in traffic scenes

**Tech Stack:** PyTorch, Torchvision, OpenCV

---

### 2. Accident Detection System (`accidentDetection.py`)

* Prototype system for detecting traffic anomalies
* Uses frame differencing and background subtraction
* Simulated trigger displays **"Accident happened"**
* Includes a logic stub for WhatsApp alerts using `pywhatkit`

**Input:** `accident.mp4`

---

### 3. Lane Line Detection (`laneDetection.py`)

* Detects and highlights lane markings on roads
* Converts frames to HSV color space for color masking
* Applies Canny Edge Detection and Hough Transform

**Input:** `road_car_view.mp4`

---

### 4. Collision Warning System (`cardetect.py`)

* Monitors distances between moving vehicles
* Uses MOG2 background subtraction for motion detection
* Calculates Euclidean distance between object centroids
* Triggers **"Collision Warning!"** when vehicles get too close

**Input:** `vi.mp4`

---

### 5. Basic Car Detection (`carDetection.py`)

* Lightweight vehicle detection using classical CV techniques
* Uses BackgroundSubtractorMOG2
* Applies morphological operations (dilation, closing)

**Input:** `video.mp4`

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

**Libraries used:**

* `opencv-python` – Image processing and video capture
* `numpy` – Numerical operations
* `torch`, `torchvision` – Deep learning (Faster R-CNN)
* `pywhatkit` – Optional WhatsApp alert integration

---

## 🚀 Usage

Ensure the required video files are available in the project directory, or update the file paths in the scripts:

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

Press **q** or **Esc** (depending on the script) to close the video window.

---

## 🔮 Future Improvements

* Real-time webcam and IP camera integration
* Enable real WhatsApp/SMS alerts in accident detection
* Vehicle speed estimation using frame displacement
* Object tracking with Kalman Filters
* Maintain consistent vehicle IDs across frames

---

## 📝 License

This project is open-source. Feel free to use, modify, and improve the code.
