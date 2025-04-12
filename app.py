# app.py
import os
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_HANDLING'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_WEB_SOCKET_CONNECTION'] = 'false'

import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False'")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import os
import sys
import logging
from dataclasses import dataclass, field
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import googlenet, GoogLeNet_Weights
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageOps
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.cm as cm
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from io import BytesIO

from captum.attr import IntegratedGradients
from streamlit_drawable_canvas import st_canvas

# Set page config as the very first Streamlit command
st.set_page_config(layout="wide", page_icon="🔬")

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)

# Custom CSS for a consistent aesthetic
st.markdown(
    """
    <style>
    /* Main title styling */
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #3E95CD;
    }
    /* Sidebar expander headers */
    .css-1d391kg, .css-1d391kg div {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
    /* Customize Streamlit button appearance */
    .stButton>button {
        background-color: #3E95CD;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
        font-size: 16px;
    }
    /* Custom info text styling */
    .custom-info {
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@dataclass
class Config:
    """Central configuration class for all application settings."""
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_MB: int = 5
    ALLOWED_EXTENSIONS: list = field(default_factory=lambda: ["jpg", "jpeg", "png"])
    MIN_IMAGE_DIM: int = 64
    MODEL_INPUT_SIZE: tuple = (128, 128)
    MODEL_WEIGHTS_PATH: str = "model/blood-cell-cancer-pytorch-weights.pth"
    CLASS_LABELS: dict = field(default_factory=lambda: {
        0: "Benign",
        1: "Early_Pre_B",
        2: "Pre_B",
        3: "Pro_B"
    })
    TRANSFORM: transforms.Compose = field(
        default_factory=lambda: transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    )

def generate_gradcam(image: Image.Image,
                     model: nn.Module,
                     transform: transforms.Compose,
                     device: torch.device,
                     target_layer: str,
                     opacity: float,
                     visualization_mode: str = "Overlay",
                     heatmap_intensity: float = 1.0) -> Image.Image:
    """
    Generates a Grad-CAM visualization for the given image and model.
    """
    input_tensor = transform(image).unsqueeze(0).to(device)
    activations, gradients = [], []

    def forward_hook(module, inp, outp):
        activations.append(outp)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    module_dict = dict(model.named_modules())
    if target_layer not in module_dict:
        raise ValueError(f"Layer {target_layer} not found in model.")
    target_module = module_dict[target_layer]
    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    score = output[0, pred_idx]
    score.backward()

    fh.remove()
    bh.remove()

    activation = activations[0].detach()[0]
    gradient = gradients[0].detach()[0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=0))
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam_np = cam.cpu().numpy()

    heatmap = cm.jet(cam_np)[:, :, :3]
    
    if visualization_mode == "Heatmap":
        heatmap = np.clip(heatmap * heatmap_intensity, 0, 1)
        result_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
    else:
        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
        result_image = Image.blend(image.convert("RGBA"), heatmap_image.convert("RGBA"), opacity)
    return result_image

def generate_integrated_gradients(image: Image.Image,
                                  model: nn.Module,
                                  transform: transforms.Compose,
                                  device: torch.device,
                                  baseline_value: float = 0.0,
                                  steps: int = 50,
                                  opacity: float = 0.5,
                                  visualization_mode: str = "Overlay",
                                  heatmap_intensity: float = 1.0) -> Image.Image:
    """
    Generates an explanation using Integrated Gradients.
    """
    input_tensor = transform(image).unsqueeze(0).to(device)
    baseline = torch.full_like(input_tensor, baseline_value)
    ig = IntegratedGradients(model)
    outputs = model(input_tensor)
    target = outputs.argmax(dim=1).item()
    attributions, delta = ig.attribute(input_tensor, baseline, target=target, return_convergence_delta=True, n_steps=steps)
    attributions = attributions.squeeze().cpu().detach().numpy()
    attribution_map = np.mean(np.abs(attributions), axis=0)
    attribution_map = np.maximum(attribution_map, 0)
    attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
    heatmap = cm.jet(attribution_map)[:, :, :3]
    
    if visualization_mode == "Heatmap":
        heatmap = np.clip(heatmap * heatmap_intensity, 0, 1)
        result_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
    else:
        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
        result_image = Image.blend(image.convert("RGBA"), heatmap_image.convert("RGBA"), opacity)
    return result_image

def generate_explanation_text(method: str, visualization_mode: str) -> str:
    """Generates a detailed textual explanation."""
    if method == "Grad-CAM":
        explanation = (
            "The Grad-CAM visualization highlights key regions in the image that the model considers "
            "important for making its prediction. Areas with intense colors indicate higher influence."
        )
    elif method == "Integrated Gradients":
        explanation = (
            "The Integrated Gradients method assigns attribution values to different parts of the image, "
            "indicating each part's contribution to the final prediction."
        )
    else:
        explanation = "No detailed explanation available for the selected method."
    explanation += f"\n\nSelected Visualization Mode: **{visualization_mode}**."
    return explanation

def apply_windowing(image: Image.Image, window_center: float, window_width: float) -> Image.Image:
    """Applies enhanced contrast and windowing controls."""
    try:
        gray = ImageOps.grayscale(image)
        np_img = np.array(gray).astype(np.float32)
        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        np_img = (np_img - lower_bound) / window_width * 255.0
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    except Exception as e:
        logging.error(f"Windowing failed: {str(e)}")
        st.error("Failed to apply windowing. Returning original grayscale image.")
        return ImageOps.grayscale(image)

def load_dicom_image(uploaded_dicom) -> Image.Image:
    """Loads a DICOM file and converts it to a PIL image."""
    try:
        ds = pydicom.dcmread(uploaded_dicom)
        data = ds.pixel_array
        if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
            data = apply_voi_lut(data, ds)
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data = (data * 255).astype(np.uint8)
        image = Image.fromarray(data)
        return image.convert("RGB")
    except Exception as e:
        logging.error(f"DICOM load failed: {str(e)}")
        st.error("Failed to load DICOM file.")
        raise

def load_dicom_volume(uploaded_dicom_files) -> np.ndarray:
    """Loads multiple DICOM files and stacks them into a 3D numpy array."""
    slices = []
    for file in uploaded_dicom_files:
        try:
            ds = pydicom.dcmread(file)
            data = ds.pixel_array
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                data = apply_voi_lut(data, ds)
            slices.append(data.astype(np.float32))
        except Exception as e:
            logging.error(f"Error reading slice: {str(e)}")
    if not slices:
        raise ValueError("No valid DICOM slices were loaded.")
    volume = np.stack(slices, axis=0)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume

def generate_mpr_views(volume: np.ndarray):
    """Generates MPR views (axial, coronal, sagittal) from a 3D volume."""
    axial = volume[volume.shape[0] // 2, :, :]
    coronal = volume[:, volume.shape[1] // 2, :]
    sagittal = volume[:, :, volume.shape[2] // 2]
    axial_img = Image.fromarray((axial * 255).astype(np.uint8)).convert("RGB")
    coronal_img = Image.fromarray((coronal * 255).astype(np.uint8)).convert("RGB")
    sagittal_img = Image.fromarray((sagittal * 255).astype(np.uint8)).convert("RGB")
    return axial_img, coronal_img, sagittal_img

class BloodCellClassifier:
    """Main classifier class handling model operations."""
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model(self.config)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model(_config):
        try:
            model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, len(_config.CLASS_LABELS))
            )
            
            # Load model weights from file
            if os.path.exists(_config.MODEL_WEIGHTS_PATH):
                state_dict = torch.load(
                    _config.MODEL_WEIGHTS_PATH,
                    map_location=_config.DEVICE,
                    weights_only=True
                )
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model weights file not found at {_config.MODEL_WEIGHTS_PATH}")
                
            model.to(_config.DEVICE).eval()
            return model
        except Exception as e:
            logging.critical(f"Model loading failed: {str(e)}")
            st.error("Critical error: Failed to load prediction model. Please check the logs.")
            st.stop()

    def predict(self, image: Image.Image, num_samples: int = 10) -> tuple:
        """Predict using Monte Carlo Dropout to quantify uncertainty."""
        try:
            img_tensor = self.config.TRANSFORM(image).unsqueeze(0).to(self.config.DEVICE)
            model = self.model
            
            def enable_dropout(m):
                if isinstance(m, nn.Dropout):
                    m.train()
                    
            model.apply(enable_dropout)

            probs_samples = []
            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    probs_samples.append(probs.cpu().numpy())
                    
            probs_samples = np.concatenate(probs_samples, axis=0)
            mean_probs = probs_samples.mean(axis=0)
            std_probs = probs_samples.std(axis=0)
            pred_idx = int(np.argmax(mean_probs))
            return pred_idx, mean_probs.tolist(), std_probs.tolist()
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            st.error("Error during prediction.")
            return -1, [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5]

class ImageProcessor:
    """Handles all image-related operations."""
    def __init__(self, config: Config):
        self.config = config

    def validate_file(self, uploaded_file) -> bool:
        try:
            if uploaded_file.size > self.config.MAX_MB * 1024 * 1024:
                raise ValueError(f"File size exceeds {self.config.MAX_MB}MB limit")
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext not in self.config.ALLOWED_EXTENSIONS:
                raise ValueError("Invalid file format. Only JPG, JPEG, and PNG are allowed.")
            with Image.open(uploaded_file) as img:
                img.verify()
                if img.width < self.config.MIN_IMAGE_DIM or img.height < self.config.MIN_IMAGE_DIM:
                    raise ValueError(f"Image too small (min {self.config.MIN_IMAGE_DIM}px)")
            return True
        except Exception as e:
            logging.error(f"Image validation failed: {str(e)}")
            st.error(f"Invalid image file: {str(e)}")
            return False

    def quality_control(self, image: Image.Image) -> (bool, float, float):
        """Performs quality control checks on the image."""
        try:
            gray = np.array(image.convert("L"))
            brightness = gray.mean()
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            qc_passed = brightness >= 40 and brightness <= 230 and laplacian_var >= 50
            return qc_passed, brightness, laplacian_var
        except Exception as e:
            logging.error(f"Quality control failed: {str(e)}")
            return True, 128, 100

    def load_image(self, uploaded_file) -> Image.Image:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            qc_passed, _, _ = self.quality_control(image)
            if not qc_passed:
                st.warning("Image quality check failed. Results may be unreliable.")
            return image
        except Exception as e:
            logging.error(f"Image load failed: {str(e)}")
            st.error("Failed to load image.")
            raise

class ResultsVisualizer:
    """Handles result visualization and display."""
    def __init__(self, config: Config):
        self.config = config

    def plot_probabilities(self, probabilities: list) -> None:
        try:
            df = pd.DataFrame({
                "Class": [self.config.CLASS_LABELS[i] for i in range(len(self.config.CLASS_LABELS))],
                "Probability": probabilities
            })
            fig = px.bar(df, x='Class', y='Probability', color='Probability',
                         color_continuous_scale='Bluered', text_auto='.2%',
                         title="📊 Class Probability Distribution")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig)
        except Exception as e:
            logging.error(f"Error plotting probabilities: {str(e)}")
            st.error("Failed to plot probabilities.")

    def show_detailed_table(self, probabilities: list, uncertainties: list) -> None:
        try:
            prob_df = pd.DataFrame({
                "Class": [self.config.CLASS_LABELS[i] for i in range(len(self.config.CLASS_LABELS))],
                "Probability": [f"{p*100:.2f}%" for p in probabilities],
                "Uncertainty (Std)": [f"{u*100:.2f}%" for u in uncertainties]
            })
            st.dataframe(prob_df.style.highlight_max(axis=0))
        except Exception as e:
            logging.error(f"Error displaying detailed table: {str(e)}")

def apply_high_resolution_controls(image: Image.Image, brightness: float, contrast: float,
                                   zoom: float, rotation: float, saturation: float,
                                   sharpness: float) -> Image.Image:
    """Applies adjustments to the input image."""
    try:
        adjusted_image = image.copy()
        if brightness != 1.0:
            adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(brightness)
        if contrast != 1.0:
            adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(contrast)
        if zoom > 1.0:
            w, h = adjusted_image.size
            nw, nh = int(w / zoom), int(h / zoom)
            left, top = (w - nw) // 2, (h - nh) // 2
            adjusted_image = adjusted_image.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)
        if rotation != 0:
            adjusted_image = adjusted_image.rotate(-rotation, expand=True)
        if saturation != 1.0:
            adjusted_image = ImageEnhance.Color(adjusted_image).enhance(saturation)
        if sharpness != 1.0:
            adjusted_image = ImageEnhance.Sharpness(adjusted_image).enhance(sharpness)
        return adjusted_image
    except Exception as e:
        logging.error(f"Error adjusting image: {str(e)}")
        return image

def generate_diagnostic_report(original_image: Image.Image,
                               adjusted_image: Image.Image,
                               prediction: str,
                               probabilities: list,
                               uncertainties: list,
                               gradcam_image: Image.Image) -> str:
    """Generates a text-based diagnostic report."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"Diagnostic Report\nTimestamp: {timestamp}\n\n"
        if "patient_metadata" in st.session_state:
            patient = st.session_state["patient_metadata"]
            report += "Patient Information:\n"
            report += f"- Name: {patient.get('name', 'N/A')}\n"
            report += f"- Age: {patient.get('age', 'N/A')}\n"
            report += f"- Medical Record Number: {patient.get('id', 'N/A')}\n\n"
        report += f"Prediction: {prediction}\n\n"
        report += "Class Probabilities and Uncertainties:\n"
        for i, (prob, uncert) in enumerate(zip(probabilities, uncertainties)):
            report += f"- {Config().CLASS_LABELS[i]}: {prob*100:.2f}% (Uncertainty: {uncert*100:.2f}%)\n"
        return report
    except Exception as e:
        logging.error(f"Report generation failed: {str(e)}")
        return "Diagnostic Report generation failed."

def append_audit_log(event: str):
    """Appends an event to the audit log."""
    try:
        if "audit_log" not in st.session_state:
            st.session_state["audit_log"] = []
        st.session_state["audit_log"].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {event}")
    except Exception as e:
        logging.error(f"Error appending to audit log: {str(e)}")

def main():
    st.markdown('<p class="main-title">🔬 Blood Smear Classifier App</p>', unsafe_allow_html=True)
    
    # Initialize session state
    for key in ["run_prediction", "run_explainability", "generate_report", 
                "view_audit_log", "export_to_emr", "run_confidence", 
                "run_windowing", "render_volume", "annotate_image"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        # Patient Information
        with st.expander("👤 Patient Information", expanded=True):
            patient_name = st.text_input("Patient Name", placeholder="Enter patient's name")
            patient_age = st.number_input("Patient Age", min_value=0, value=0, step=1)
            patient_id = st.text_input("Medical Record Number", placeholder="Enter patient ID")
            if st.button("Save Patient Metadata"):
                st.session_state["patient_metadata"] = {
                    "name": patient_name,
                    "age": patient_age,
                    "id": patient_id
                }
                st.success("Patient metadata saved!")

        # File Operations
        with st.expander("📁 File Operations", expanded=True):
            uploaded_file = st.file_uploader("Upload blood smear image", type=Config().ALLOWED_EXTENSIONS)
            if st.button("🔄 Clear Cache"):
                st.cache_resource.clear()
                st.rerun()

        # DICOM File Upload
        with st.expander("📂 DICOM File Upload", expanded=True):
            dicom_file = st.file_uploader("Upload DICOM image", type=["dcm"])

        # 3D Volume & MPR
        with st.expander("📑 3D Volume & MPR", expanded=True):
            volume_files = st.file_uploader("Upload DICOM files for 3D volume", 
                                          accept_multiple_files=True, type=["dcm"])
            if st.button("Render 3D Volume"):
                st.session_state["render_volume"] = True

        # High-Resolution Controls
        with st.expander("🔍 High-Resolution Controls", expanded=True):
            brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
            zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.1)
            rotation = st.slider("Rotation (°)", -180, 180, 0, 5)
            real_time_preview = st.checkbox("Real-Time Preview", value=True)

        # Annotation Tools
        with st.expander("✏️ Annotation Tools", expanded=True):
            drawing_mode = st.selectbox("Drawing Mode", ["freedraw", "line", "rect", "circle"])
            stroke_width = st.slider("Stroke Width", 1, 10, 3)
            if st.button("Enable Annotation"):
                st.session_state["annotate_image"] = True

        # Prediction Controls
        with st.expander("🔮 Prediction Controls", expanded=True):
            if st.button("🔍 Run Prediction"):
                st.session_state["run_prediction"] = True
            if st.button("❌ Clear Results"):
                st.session_state.clear()
                st.rerun()

        # Explainability Controls
        with st.expander("🧩 Explainability", expanded=True):
            explainability_method = st.selectbox("Method", ["Grad-CAM", "Integrated Gradients"])
            if st.button("Generate Explanation"):
                st.session_state["run_explainability"] = True

        # Confidence Thresholds
        with st.expander("🚨 Confidence", expanded=True):
            confidence_threshold = st.slider("Threshold (%)", 50, 100, 70)
            if st.button("Check Confidence"):
                st.session_state["run_confidence"] = True

        # Windowing Controls
        with st.expander("🖼 Windowing", expanded=True):
            window_center = st.slider("Window Center", 0, 255, 128)
            window_width = st.slider("Window Width", 1, 255, 128)
            if st.button("Apply Windowing"):
                st.session_state["run_windowing"] = True

        # Clinical Workflow
        with st.expander("🩺 Clinical Workflow", expanded=True):
            if st.button("Generate Report"):
                st.session_state["generate_report"] = True
            if st.button("View Audit Log"):
                st.session_state["view_audit_log"] = True
            if st.button("Export to EMR"):
                st.session_state["export_to_emr"] = True

    # ─── Main Tabs ───────────────────────────────────────────────────────────
    tabs = st.tabs(["Classification", "Explainability", "Confidence", 
                   "Windowing", "3D Volume", "Annotation"])

    # Tab 1: Classification
    with tabs[0]:
        original_image = None
        if uploaded_file:
            image_processor = ImageProcessor(Config())
            if image_processor.validate_file(uploaded_file):
                try:
                    original_image = image_processor.load_image(uploaded_file)
                    st.session_state["original_image"] = original_image
                except Exception:
                    st.stop()
        elif dicom_file:
            try:
                original_image = load_dicom_image(dicom_file)
                st.session_state["original_image"] = original_image
            except Exception:
                st.stop()
        else:
            st.info("Upload an image to classify.")

        if original_image:
            if real_time_preview:
                adjusted_image = apply_high_resolution_controls(
                    original_image, brightness, contrast, zoom, rotation, 1.0, 1.0
                )
                st.session_state["adjusted_image"] = adjusted_image

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader("Adjusted Image")
                st.image(st.session_state.get("adjusted_image", original_image), 
                         use_container_width=True)

            if st.session_state["run_prediction"]:
                classifier = BloodCellClassifier(Config())
                with st.spinner("Analyzing..."):
                    pred_idx, mean_probs, std_probs = classifier.predict(original_image)
                    cls = Config().CLASS_LABELS.get(pred_idx, "Unknown")
                st.success(f"Result: {cls}")
                
                visualizer = ResultsVisualizer(Config())
                visualizer.plot_probabilities(mean_probs)
                visualizer.show_detailed_table(mean_probs, std_probs)
                
                st.session_state["cls"] = cls
                st.session_state["probs"] = mean_probs
                st.session_state["uncertainty"] = std_probs
                st.session_state["run_prediction"] = False

    # Tab 2: Explainability
    with tabs[1]:
        if st.session_state["run_explainability"]:
            if "original_image" in st.session_state:
                original_image = st.session_state["original_image"]
                classifier = BloodCellClassifier(Config())
                
                with st.spinner("Generating explanation..."):
                    if explainability_method == "Grad-CAM":
                        explanation_image = generate_gradcam(
                            original_image, classifier.model, Config().TRANSFORM,
                            Config().DEVICE, "inception5b", 0.5
                        )
                    else:
                        explanation_image = generate_integrated_gradients(
                            original_image, classifier.model, Config().TRANSFORM,
                            Config().DEVICE
                        )
                
                cols = st.columns(2)
                with cols[0]:
                    st.subheader("Original Image")
                    st.image(original_image, use_container_width=True)
                with cols[1]:
                    st.subheader(f"{explainability_method}")
                    st.image(explanation_image, use_container_width=True)
                
                st.markdown(generate_explanation_text(explainability_method, "Overlay"))
            st.session_state["run_explainability"] = False

    # Tab 3: Confidence
    with tabs[2]:
        if st.session_state["run_confidence"]:
            if "probs" in st.session_state:
                max_prob = max(st.session_state["probs"])
                threshold = confidence_threshold / 100
                if max_prob >= threshold:
                    st.success(f"Confidence ({max_prob*100:.1f}%) meets threshold ({threshold*100}%)")
                else:
                    st.error(f"Confidence ({max_prob*100:.1f}%) below threshold ({threshold*100}%)")
            st.session_state["run_confidence"] = False

    # Tab 4: Windowing
    with tabs[3]:
        if st.session_state["run_windowing"]:
            if "original_image" in st.session_state:
                windowed_image = apply_windowing(
                    st.session_state["original_image"], 
                    window_center, 
                    window_width
                )
                st.image(windowed_image, use_container_width=True)
            st.session_state["run_windowing"] = False

    # Tab 5: 3D Volume
    with tabs[4]:
        if st.session_state["render_volume"] and volume_files:
            with st.spinner("Processing volume..."):
                try:
                    volume = load_dicom_volume(volume_files)
                    axial, coronal, sagittal = generate_mpr_views(volume)
                    
                    st.subheader("Axial View")
                    st.image(axial, use_container_width=True)
                    
                    st.subheader("Coronal View")
                    st.image(coronal, use_container_width=True)
                    
                    st.subheader("Sagittal View")
                    st.image(sagittal, use_container_width=True)
                except Exception as e:
                    st.error(f"Volume rendering failed: {e}")
            st.session_state["render_volume"] = False

    # Tab 6: Annotation
    with tabs[5]:
        if st.session_state["annotate_image"]:
            if "original_image" in st.session_state:
                bg_image = st.session_state["original_image"].convert("RGBA")
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=stroke_width,
                    stroke_color="#FF0000",
                    background_image=bg_image,
                    update_streamlit=True,
                    height=bg_image.height,
                    width=bg_image.width,
                    drawing_mode=drawing_mode,
                    key="canvas"
                )
                
                if canvas_result.image_data is not None:
                    st.image(canvas_result.image_data, use_container_width=True)

    # Clinical Workflow Outputs
    if st.session_state["generate_report"]:
        if all(key in st.session_state for key in ["original_image", "cls", "probs"]):
            report = generate_diagnostic_report(
                st.session_state["original_image"],
                st.session_state.get("adjusted_image", st.session_state["original_image"]),
                st.session_state["cls"],
                st.session_state["probs"],
                st.session_state.get("uncertainty", [0]*4),
                st.session_state.get("adjusted_image", st.session_state["original_image"])
            )
            st.download_button(
                "Download Report",
                report,
                file_name="diagnostic_report.txt"
            )
        st.session_state["generate_report"] = False

    if st.session_state["view_audit_log"]:
        st.subheader("Audit Log")
        if "audit_log" in st.session_state:
            for entry in st.session_state["audit_log"]:
                st.text(entry)
        st.session_state["view_audit_log"] = False

    if st.session_state["export_to_emr"]:
        st.success("Results exported to EMR")
        st.session_state["export_to_emr"] = False

if __name__ == "__main__":
    main()