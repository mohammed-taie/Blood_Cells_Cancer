# 🧠 Medical Image AI Viewer

A powerful Streamlit-based medical imaging web app for AI-assisted diagnosis, visualization, and image annotation. This tool is designed for radiologists, healthcare professionals, and researchers to interactively explore and analyze medical images with support for 3D MPR views, AI prediction, Grad-CAM overlays, confidence alerts, DICOM volume loading, windowing adjustments, and more.

---

## 🚀 Features

- ✅ **AI Prediction** with class probabilities and uncertainty estimates
- 📊 **Confidence Alerts** with threshold-based risk flagging
- 🖼️ **Image Viewer** with original and windowed views
- 🧭 **Windowing Controls** (center/width sliders for grayscale contrast)
- 🧩 **3D Volume Rendering** from DICOM files (Axial, Coronal, Sagittal MPR)
- ✏️ **Interactive Annotation Tools** (freedraw, rectangle, circle, line)
- 🧾 **Diagnostic Report Generation** (downloadable as `.txt`)
- 🕵️ **Audit Logging** for clinical traceability
- 🔄 **EMR Export Simulation**

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/medical-image-ai-viewer.git
cd medical-image-ai-viewer
pip install -r requirements.txt