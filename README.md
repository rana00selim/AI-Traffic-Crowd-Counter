<div align="center">

# 👁️ AI Traffic & Crowd Analytics

**Real-Time Vehicle & Pedestrian Intelligence with YOLOv8 & Computer Vision**

Smart city, traffic engineering, and public safety analytics powered by modern deep learning.

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

<br/>

<img src="https://via.placeholder.com/900x450?text=Demo+GIF+Coming+Soon" alt="Demo Preview" width="900"/>

<br/>

[Overview](#-overview) •
[Features](#-key-features) •
[Tech Stack](#-tech-stack) •
[Installation](#-installation) •
[Usage](#-usage) •
[Project Structure](#-project-structure) •
[Roadmap](#-roadmap) •
[Author](#-author)

</div>

---

## 📚 Overview

**AI Traffic & Crowd Analytics** is an end-to-end **real-time computer vision dashboard** designed to analyze vehicle flow and pedestrian behavior using modern deep learning techniques.

Built on top of **YOLOv8**, **OpenCV**, and **Streamlit**, the system goes beyond basic object detection by providing:

- Persistent object tracking
- Line-based and zone-based counting
- Interactive analytics dashboards
- Real-world smart city use cases

This project is suitable for:

- 🚦 Smart Traffic Management
- 🏙️ Urban Analytics & City Planning
- 🛍️ Retail Footfall Analysis
- 🚨 Public Safety & Crowd Monitoring

---

## 🧩 Key Features

| Feature                    | Description                                                                                      | Classes                             |
| -------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------- |
| 🚗 **Vehicle Counting**    | Counts vehicles crossing a configurable virtual line with ID-based tracking to avoid duplicates. | `Car`, `Bus`, `Truck`, `Motorcycle` |
| 🚶 **Pedestrian Counting** | Optimized logic for dense pedestrian movement and direction-aware counting.                      | `Person`                            |
| 👥 **Zone Occupancy**      | Tracks how many objects are inside a defined ROI in real-time. Objects are visually color-coded. | `Person`, `Vehicle`                 |
| 🎯 **ID Tracking**         | Persistent IDs via ByteTrack ensure stable counting across frames.                               | All                                 |
| 📊 **Live Analytics**      | Dynamic charts update per frame using Plotly.                                                    | Metrics                             |

---

## 🛠️ Tech Stack

<div align="center">

| Layer             | Technology              | Purpose                      |
| ----------------- | ----------------------- | ---------------------------- |
| **Language**      | Python 3.8+             | Core logic & orchestration   |
| **Model**         | YOLOv8 (Ultralytics)    | Object detection             |
| **Tracking**      | Supervision + ByteTrack | Multi-object tracking        |
| **UI**            | Streamlit               | Interactive dashboard        |
| **Visualization** | OpenCV, Plotly          | Frame annotation & analytics |
| **Data**          | NumPy, Pandas           | Metrics handling             |

</div>

---

## 🚀 Installation

### 1️⃣ Prerequisites

- Python **3.8 or higher**
- Git
- (Optional) CUDA-enabled GPU for better performance

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/username/AI-Traffic-Counter.git
cd AI-Traffic-Counter
```

### 3️⃣ Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate:**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

Start the interactive dashboard with:

```bash
streamlit run dashboard.py
```

The app will launch automatically in your browser.

### 🎛️ Dashboard Controls

**🔹 Sidebar Options**

- **Task Mode**: Vehicle Counting, Person Counting, Zone Occupancy
- **YOLO Model Size**: Nano → Fast, Small/Medium, Large → High Accuracy
- **Input Source**: Video file (.mp4), Live webcam feed

**🔹 Geometry Configuration**

- Adjust counting lines and ROI zones visually
- Fine-tune positioning for different camera angles
- Real-time visual feedback on the video stream

**🔹 Analytics Panel**

- Live count updates
- Frame-by-frame statistics
- Interactive Plotly charts

## 📂 Project Structure

```
📦 AI-Traffic-Counter
 ┣ 📂 assets              # Images, demo GIFs, screenshots
 ┣ 📂 models              # YOLOv8 .pt models
 ┣ 📜 dashboard.py        # Streamlit UI + App Logic
 ┣ 📜 vehicle_counter.py  # Detection, tracking & counting logic
 ┣ 📜 requirements.txt    # Python dependencies
 ┗ 📜 README.md           # Documentation
```

## 🔮 Roadmap

- License Plate Recognition (LPR)
- Historical Data Export (CSV / Excel)
- RTSP & CCTV Stream Support
- Multi-Camera Dashboard
- Cloud Deployment (Docker + Streamlit Cloud)

## 👩‍💻 Author

**Dr. Murat Altun | Rana Selim**  
Computer Engineer • AI & Computer Vision Enthusiast

📫 Contributions, feedback, and ideas are always welcome.
