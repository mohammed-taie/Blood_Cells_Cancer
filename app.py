# app.py - Complete Blood Cell Classifier for Streamlit Sharing

import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

import os
import io
import sys
import logging
import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Dict, Optional

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
from captum.attr import IntegratedGradients
from streamlit_drawable_canvas import st_canvas

# Configuration Constants
STREAMLIT_SHARING = True
MAX_FILE_SIZE_MB = 5
MODEL_WEIGHTS_PATH = "model/blood-cell-cancer-pytorch-weights.pth"

# Initialize Streamlit
st.set_page_config(layout="wide", page_icon="🔬")

# Custom CSS
st.markdown("""
<style>
.main-title { font-size: 32px; font-weight: bold; color: #3E95CD; }
.stButton>button { background-color: #3E95CD; color: white; border: none; padding: 0.5em 1em; border-radius: 5px; }
.custom-info { color: #6c757d; }
.stAlert { padding: 0.5rem; }
.stProgress > div > div { background-color: #3E95CD; }
</style>
""", unsafe_allow_html=True)

@dataclass
class Config:
    """Application configuration"""
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_MB: int = MAX_FILE_SIZE_MB
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "dcm"])
    MIN_IMAGE_DIM: int = 64
    MODEL_INPUT_SIZE: Tuple[int, int] = (128, 128)
    MODEL_WEIGHTS_PATH: str = MODEL_WEIGHTS_PATH
    CLASS_LABELS: Dict[int, str] = field(default_factory=lambda: {
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

def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    """Generate download link for PIL images"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'

def generate_gradcam(image: Image.Image,
                    model: nn.Module,
                    transform: transforms.Compose,
                    device: torch.device,
                    target_layer: str,
                    opacity: float,
                    visualization_mode: str = "Overlay",
                    heatmap_intensity: float = 1.0) -> Image.Image:
    """Generate Grad-CAM visualization"""
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        activations, gradients = [], []

        def forward_hook(module, inp, outp):
            activations.append(outp.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        module_dict = dict(model.named_modules())
        if target_layer not in module_dict:
            st.error(f"Layer {target_layer} not found in model")
            return image
            
        target_module = module_dict[target_layer]
        
        with torch.no_grad():
            fh = target_module.register_forward_hook(forward_hook)
            bh = target_module.register_full_backward_hook(backward_hook)

            model.zero_grad()
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            score = output[0, pred_idx]
            score.backward()

            fh.remove()
            bh.remove()

        activation = activations[0][0]
        gradient = gradients[0][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_np = cam.cpu().numpy()

        heatmap = cm.jet(cam_np)[:, :, :3]
        
        if visualization_mode == "Heatmap":
            heatmap = np.clip(heatmap * heatmap_intensity, 0, 1)
            result_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
        else:
            heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.LANCZOS)
            result_image = Image.blend(image.convert("RGBA"), heatmap_image.convert("RGBA"), opacity)
        return result_image
    except Exception as e:
        st.error(f"Grad-CAM failed: {str(e)}")
        return image

def generate_integrated_gradients(image: Image.Image,
                                 model: nn.Module,
                                 transform: transforms.Compose,
                                 device: torch.device,
                                 baseline_value: float = 0.0,
                                 steps: int = 50,
                                 opacity: float = 0.5,
                                 visualization_mode: str = "Overlay",
                                 heatmap_intensity: float = 1.0) -> Image.Image:
    """Generate Integrated Gradients visualization"""
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        baseline = torch.full_like(input_tensor, baseline_value)
        ig = IntegratedGradients(model)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            target = outputs.argmax(dim=1).item()
            attributions, delta = ig.attribute(input_tensor, baseline, target=target, 
                                             return_convergence_delta=True, n_steps=steps)
        
        attributions = attributions.squeeze().cpu().numpy()
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
    except Exception as e:
        st.error(f"Integrated Gradients failed: {str(e)}")
        return image

def generate_explanation_text(method: str, visualization_mode: str) -> str:
    """Generate explanation text for the selected method"""
    explanations = {
        "Grad-CAM": (
            "The Grad-CAM visualization highlights key regions in the image that the model considers "
            "important for its prediction. Warmer colors indicate higher influence. "
            "Overlay mode shows the heatmap superimposed on the original image, while "
            "Heatmap mode shows only the activation regions."
        ),
        "Integrated Gradients": (
            "Integrated Gradients attributes the prediction to input features by integrating "
            "gradients from a baseline. The heatmap shows which pixels contributed most to "
            "the prediction, with brighter areas having more influence."
        )
    }
    return f"{explanations.get(method, 'No explanation available')}\n\nMode: {visualization_mode}"

def apply_windowing(image: Image.Image, window_center: float, window_width: float) -> Image.Image:
    """Apply DICOM-style windowing to image"""
    try:
        gray = ImageOps.grayscale(image)
        np_img = np.array(gray).astype(np.float32)
        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        np_img = (np_img - lower_bound) / window_width * 255.0
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    except Exception as e:
        st.error(f"Windowing failed: {str(e)}")
        return ImageOps.grayscale(image)

def load_dicom_image(uploaded_file) -> Image.Image:
    """Load DICOM file and convert to PIL Image with proper file handling"""
    try:
        # Ensure we're reading from the start of the file
        uploaded_file.seek(0)
        
        # Read DICOM file
        ds = pydicom.dcmread(uploaded_file)
        data = ds.pixel_array
        
        # Apply VOI LUT if available
        if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
            data = apply_voi_lut(data, ds)
            
        # Normalize and convert to 8-bit
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data = (data * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(data)
        return image.convert("RGB")
    except Exception as e:
        st.error(f"DICOM load failed: {str(e)}")
        raise

def load_dicom_volume(uploaded_files) -> np.ndarray:
    """Load multiple DICOM files into 3D volume"""
    slices = []
    for file in uploaded_files:
        try:
            ds = pydicom.dcmread(file)
            data = ds.pixel_array
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                data = apply_voi_lut(data, ds)
            slices.append(data.astype(np.float32))
        except Exception as e:
            st.error(f"Error reading slice: {str(e)}")
    
    if not slices:
        raise ValueError("No valid DICOM slices loaded")
    
    try:
        slices = sorted(slices, key=lambda s: s.InstanceNumber)
    except:
        pass
    
    volume = np.stack(slices, axis=0)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume

def generate_mpr_views(volume: np.ndarray) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Generate MPR views from 3D volume"""
    axial = volume[volume.shape[0] // 2, :, :]
    coronal = volume[:, volume.shape[1] // 2, :]
    sagittal = volume[:, :, volume.shape[2] // 2]
    
    axial_img = Image.fromarray((axial * 255).astype(np.uint8)).convert("RGB")
    coronal_img = Image.fromarray((coronal * 255).astype(np.uint8)).convert("RGB")
    sagittal_img = Image.fromarray((sagittal * 255).astype(np.uint8)).convert("RGB")
    
    return axial_img, coronal_img, sagittal_img

class BloodCellClassifier:
    """Main classifier class"""
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
            
            if os.path.exists(_config.MODEL_WEIGHTS_PATH):
                state_dict = torch.load(
                    _config.MODEL_WEIGHTS_PATH,
                    map_location=_config.DEVICE,
                    weights_only=True
                )
                model.load_state_dict(state_dict)
            else:
                st.warning("Using ImageNet-pretrained base with random head")
                
            model.to(_config.DEVICE).eval()
            return model
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.stop()

    def predict(self, image: Image.Image, num_samples: int = 5) -> Tuple[int, List[float], List[float]]:
        """Make prediction with uncertainty estimation"""
        try:
            img_tensor = self.config.TRANSFORM(image).unsqueeze(0).to(self.config.DEVICE)
            model = self.model
            
            def enable_dropout(m):
                if isinstance(m, nn.Dropout):
                    m.train()
                    
            model.apply(enable_dropout)
            
            with torch.no_grad():
                probs_samples = []
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
            st.error(f"Prediction failed: {str(e)}")
            fallback_probs = [1.0 / len(self.config.CLASS_LABELS)] * len(self.config.CLASS_LABELS)
            fallback_uncertainty = [0.5] * len(self.config.CLASS_LABELS)
            return -1, fallback_probs, fallback_uncertainty

class ImageProcessor:
    """Handles image processing and validation"""
    def __init__(self, config: Config):
        self.config = config

    def validate_file(self, uploaded_file) -> bool:
        """Validate both regular images and DICOM files"""
        try:
            # Check file size
            if uploaded_file.size > self.config.MAX_MB * 1024 * 1024:
                raise ValueError(f"File exceeds {self.config.MAX_MB}MB limit")
                
            # Check file extension
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext not in self.config.ALLOWED_EXTENSIONS:
                raise ValueError(f"Invalid file type. Allowed: {self.config.ALLOWED_EXTENSIONS}")
            
            # Special handling for DICOM files
            if ext == 'dcm':
                try:
                    # Just verify we can read the DICOM header
                    uploaded_file.seek(0)
                    pydicom.dcmread(uploaded_file, stop_before_pixels=True)
                    return True
                except Exception as e:
                    raise ValueError(f"Invalid DICOM file: {str(e)}")
            else:
                # For regular images, use PIL to verify
                with Image.open(uploaded_file) as img:
                    img.verify()
                    if min(img.size) < self.config.MIN_IMAGE_DIM:
                        raise ValueError(f"Image too small (min {self.config.MIN_IMAGE_DIM}px)")
                return True
        except Exception as e:
            st.error(f"Invalid file: {str(e)}")
            return False

    def quality_control(self, image: Image.Image) -> Tuple[bool, float, float]:
        """Perform basic quality checks on image"""
        try:
            gray = np.array(image.convert("L"))
            brightness = gray.mean()
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            qc_passed = (40 <= brightness <= 230) and (laplacian_var >= 50)
            return qc_passed, brightness, laplacian_var
        except Exception as e:
            st.error(f"Quality check failed: {str(e)}")
            return False, 0, 0

    def load_image(self, uploaded_file) -> Image.Image:
        """Load either regular image or DICOM file"""
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            
            if ext == 'dcm':
                # Handle DICOM file
                uploaded_file.seek(0)  # Rewind the file
                return load_dicom_image(uploaded_file)
            else:
                # Handle regular image
                image = Image.open(uploaded_file).convert("RGB")
                qc_passed, _, _ = self.quality_control(image)
                if not qc_passed:
                    st.warning("Image quality check failed - proceed with caution")
                return image
        except Exception as e:
            st.error(f"Image load failed: {str(e)}")
            raise

class ResultsVisualizer:
    """Handles visualization of results"""
    def __init__(self, config: Config):
        self.config = config

    def plot_probabilities(self, probabilities: List[float]) -> None:
        """Plot class probabilities"""
        try:
            df = pd.DataFrame({
                "Class": [self.config.CLASS_LABELS[i] for i in range(len(self.config.CLASS_LABELS))],
                "Probability": probabilities
            })
            fig = px.bar(df, x='Class', y='Probability', color='Probability',
                         color_continuous_scale='Bluered', text_auto='.2%',
                         title="Class Probability Distribution")
            fig.update_layout(yaxis_tickformat=".0%",
                            xaxis_title="Cell Type",
                            yaxis_title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to plot probabilities: {str(e)}")

    def show_detailed_table(self, probabilities: List[float], uncertainties: List[float]) -> None:
        """Show detailed results table"""
        try:
            df = pd.DataFrame({
                "Class": [self.config.CLASS_LABELS[i] for i in range(len(self.config.CLASS_LABELS))],
                "Probability (%)": [f"{p*100:.2f}" for p in probabilities],
                "Uncertainty (%)": [f"{u*100:.2f}" for u in uncertainties]
            })
            st.dataframe(df.style.highlight_max(axis=0, subset=["Probability (%)"]))
        except Exception as e:
            st.error(f"Failed to create table: {str(e)}")

def apply_high_resolution_controls(image: Image.Image,
                                  brightness: float,
                                  contrast: float,
                                  zoom: float,
                                  rotation: float,
                                  saturation: float,
                                  sharpness: float) -> Image.Image:
    """Apply image adjustments"""
    try:
        adjusted = image.copy()
        if brightness != 1.0:
            adjusted = ImageEnhance.Brightness(adjusted).enhance(brightness)
        if contrast != 1.0:
            adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
        if zoom > 1.0:
            w, h = adjusted.size
            nw, nh = int(w / zoom), int(h / zoom)
            left, top = (w - nw) // 2, (h - nh) // 2
            adjusted = adjusted.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)
        if rotation != 0:
            adjusted = adjusted.rotate(-rotation, expand=True)
        if saturation != 1.0:
            adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
        if sharpness != 1.0:
            adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)
        return adjusted
    except Exception as e:
        st.error(f"Image adjustment failed: {str(e)}")
        return image

def generate_diagnostic_report(original_image: Image.Image,
                              adjusted_image: Image.Image,
                              prediction: str,
                              probabilities: List[float],
                              uncertainties: List[float],
                              gradcam_image: Image.Image) -> str:
    """Generate diagnostic report"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"Diagnostic Report\nTimestamp: {timestamp}\n\n"
        
        if "patient_metadata" in st.session_state:
            patient = st.session_state["patient_metadata"]
            report += "Patient Information:\n"
            report += f"- Name: {patient.get('name', 'N/A')}\n"
            report += f"- Age: {patient.get('age', 'N/A')}\n"
            report += f"- Medical Record Number: {patient.get('id', 'N/A')}\n"
            report += f"- Clinical Notes: {patient.get('notes', 'N/A')}\n\n"
        
        report += f"Prediction: {prediction}\n\n"
        report += "Class Probabilities and Uncertainties:\n"
        for i, (prob, uncert) in enumerate(zip(probabilities, uncertainties)):
            report += f"- {Config().CLASS_LABELS[i]}: {prob*100:.2f}% (Uncertainty: {uncert*100:.2f}%)\n"
        
        report += "\nNote: This report is generated by an AI-based assistive diagnostic tool and should be used alongside clinical evaluation."
        return report
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return "Diagnostic report generation failed."

def append_audit_log(event: str) -> None:
    """Add entry to audit log"""
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    st.session_state["audit_log"].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {event}")

def initialize_session_state() -> None:
    """Initialize all required session state variables"""
    required_keys = {
        'run_prediction': False,
        'run_explainability': False,
        'generate_report': False,
        'view_audit_log': False,
        'export_to_emr': False,
        'run_confidence': False,
        'run_windowing': False,
        'render_volume': False,
        'annotate_image': False,
        'original_image': None,
        'adjusted_image': None,
        'patient_metadata': None,
        'audit_log': [],
        'feedback_log': [],
        'config': Config()
    }
    
    for key, value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    initialize_session_state()
    
    st.markdown('<p class="main-title">🔬 Blood Smear Classifier App</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Patient Information
        with st.expander("👤 Patient Information", expanded=True):
            patient_info = {
                "name": st.text_input("Patient Name", ""),
                "age": st.number_input("Patient Age", 0, 120, 0),
                "id": st.text_input("Medical Record Number", ""),
                "notes": st.text_area("Clinical Notes", "", height=100)
            }
            if st.button("Save Metadata"):
                st.session_state.patient_metadata = patient_info
                st.success("Patient metadata saved!")
                append_audit_log("Patient metadata updated")
        
        # File Operations
        with st.expander("📁 File Operations", expanded=True):
            uploaded_file = st.file_uploader("Upload Image", 
                                          type=st.session_state.config.ALLOWED_EXTENSIONS,
                                          accept_multiple_files=False)
            
            dicom_file = st.file_uploader("Upload DICOM", type=["dcm"])
            volume_files = st.file_uploader("Upload DICOM Volume", type=["dcm"], accept_multiple_files=True)
            
            if st.button("Clear Cache"):
                st.cache_resource.clear()
                st.rerun()
        
        # Image Controls
        with st.expander("🔍 Image Controls", expanded=True):
            brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
            zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.1)
            rotation = st.slider("Rotation (°)", -180, 180, 0, 5)
            saturation = st.slider("Saturation", 0.5, 1.5, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 1.5, 1.0, 0.1)
            real_time_preview = st.checkbox("Real-Time Preview", True)
        
        # Windowing Controls
        with st.expander("🖼 Windowing Controls", expanded=True):
            window_center = st.slider("Window Center", 0, 255, 128)
            window_width = st.slider("Window Width", 1, 255, 128)
            if st.button("Apply Windowing"):
                st.session_state.run_windowing = True
        
        # Prediction Controls
        with st.expander("🔮 Prediction", expanded=True):
            if st.button("Run Prediction"):
                st.session_state.run_prediction = True
            confidence_threshold = st.slider("Confidence Threshold", 50, 100, 70)
            if st.button("Check Confidence"):
                st.session_state.run_confidence = True
        
        # Explainability
        with st.expander("🧩 Explainability", expanded=True):
            explain_method = st.selectbox("Method", ["Grad-CAM", "Integrated Gradients"])
            target_layer = st.text_input("Target Layer", "inception5b")
            opacity = st.slider("Opacity", 0.0, 1.0, 0.5)
            viz_mode = st.selectbox("Visualization", ["Overlay", "Heatmap"])
            if st.button("Generate Explanation"):
                st.session_state.run_explainability = True
        
        # Annotation Tools
        with st.expander("✏️ Annotation", expanded=True):
            drawing_mode = st.selectbox("Tool", ["freedraw", "line", "rect", "circle"])
            stroke_width = st.slider("Stroke", 1, 10, 3)
            stroke_color = st.color_picker("Color", "#FF0000")
            if st.button("Enable Annotation"):
                st.session_state.annotate_image = True
        
        # Clinical Workflow
        with st.expander("🩺 Workflow", expanded=True):
            if st.button("Generate Report"):
                st.session_state.generate_report = True
            if st.button("View Audit Log"):
                st.session_state.view_audit_log = True
            if st.button("Export to EMR"):
                st.session_state.export_to_emr = True
        
        # 3D Volume
        with st.expander("📦 3D Volume", expanded=True):
            if st.button("Render Volume"):
                st.session_state.render_volume = True
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Classification", "Explainability", "Confidence", 
        "Windowing", "3D Volume", "Annotations"
    ])
    
    # Tab 1: Classification
    with tab1:
        st.subheader("Image Classification")
        
        if uploaded_file or dicom_file:
            processor = ImageProcessor(st.session_state.config)
            file_to_process = dicom_file if dicom_file else uploaded_file
            
            try:
                if processor.validate_file(file_to_process):
                    if dicom_file:
                        original_image = load_dicom_image(dicom_file)
                    else:
                        original_image = processor.load_image(uploaded_file)
                    
                    st.session_state.original_image = original_image
                    
                    # Apply image adjustments
                    if real_time_preview or st.button("Update Preview"):
                        adjusted_image = apply_high_resolution_controls(
                            original_image, brightness, contrast, zoom,
                            rotation, saturation, sharpness
                        )
                        st.session_state.adjusted_image = adjusted_image
                    else:
                        adjusted_image = original_image
                    
                    # Display images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(adjusted_image, caption="Adjusted Image", use_container_width=True)
                    
                    # Quality check
                    qc_passed, qc_brightness, qc_sharpness = processor.quality_control(original_image)
                    if qc_passed:
                        st.success("Quality Check Passed")
                    else:
                        st.warning(f"Quality Issues - Brightness: {qc_brightness:.1f}, Sharpness: {qc_sharpness:.1f}")
                    
                    # Run prediction if requested
                    if st.session_state.run_prediction:
                        classifier = BloodCellClassifier(st.session_state.config)
                        with st.spinner("Analyzing..."):
                            pred_idx, probs, uncerts = classifier.predict(original_image)
                        
                        prediction = st.session_state.config.CLASS_LABELS.get(pred_idx, "Unknown")
                        st.success(f"Prediction: {prediction}")
                        st.progress(probs[pred_idx], text=f"Confidence: {probs[pred_idx]*100:.2f}%")
                        
                        visualizer = ResultsVisualizer(st.session_state.config)
                        visualizer.plot_probabilities(probs)
                        visualizer.show_detailed_table(probs, uncerts)
                        
                        st.session_state.cls = prediction
                        st.session_state.probs = probs
                        st.session_state.uncertainty = uncerts
                        append_audit_log(f"Prediction: {prediction}")
                        
                        st.session_state.run_prediction = False
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Upload an image to begin analysis")
    
    # Tab 2: Explainability
    with tab2:
        st.subheader("Model Explainability")
        
        if st.session_state.run_explainability and st.session_state.original_image:
            classifier = BloodCellClassifier(st.session_state.config)
            original_image = st.session_state.original_image
            
            with st.spinner("Generating explanation..."):
                if explain_method == "Grad-CAM":
                    explanation_image = generate_gradcam(
                        original_image, classifier.model, 
                        st.session_state.config.TRANSFORM,
                        st.session_state.config.DEVICE,
                        target_layer, opacity, viz_mode
                    )
                else:
                    explanation_image = generate_integrated_gradients(
                        original_image, classifier.model,
                        st.session_state.config.TRANSFORM,
                        st.session_state.config.DEVICE,
                        opacity=opacity,
                        visualization_mode=viz_mode
                    )
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original", use_container_width=True)
            with col2:
                st.image(explanation_image, caption="Explanation", use_container_width=True)
            
            st.markdown("### Explanation")
            st.markdown(generate_explanation_text(explain_method, viz_mode))
            
            st.session_state.run_explainability = False
        else:
            st.info("Generate an explanation from the sidebar")
    
    # [Additional tabs implementation...]
    # Tab 3: Confidence
    with tab3:
        st.subheader("Confidence Analysis")
        
        if st.session_state.run_confidence and "probs" in st.session_state:
            probs = st.session_state.probs
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            threshold = confidence_threshold / 100
            
            st.metric("Prediction Confidence", f"{confidence*100:.2f}%")
            st.metric("Set Threshold", f"{confidence_threshold}%")
            
            if confidence >= threshold:
                st.success("Confidence meets threshold")
            else:
                st.warning("Confidence below threshold - review recommended")
            
            st.session_state.run_confidence = False
        else:
            st.info("Run prediction and check confidence from sidebar")
    
    # Tab 4: Windowing
    with tab4:
        st.subheader("Image Windowing")
        
        if st.session_state.run_windowing and st.session_state.original_image:
            original = st.session_state.original_image
            windowed = apply_windowing(original, window_center, window_width)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(ImageOps.grayscale(original), caption="Original", use_container_width=True)
            with col2:
                st.image(windowed, caption="Windowed", use_container_width=True)
            
            st.session_state.run_windowing = False
        else:
            st.info("Apply windowing from sidebar controls")
    
    # Tab 5: 3D Volume
    with tab5:
        st.subheader("3D Volume Rendering")
        
        if st.session_state.render_volume and volume_files:
            try:
                with st.spinner("Loading volume..."):
                    volume = load_dicom_volume(volume_files)
                    axial, coronal, sagittal = generate_mpr_views(volume)
                
                st.image(axial, caption="Axial View", use_container_width=True)
                st.image(coronal, caption="Coronal View", use_container_width=True)
                st.image(sagittal, caption="Sagittal View", use_container_width=True)
                
                st.session_state.render_volume = False
            except Exception as e:
                st.error(f"Volume rendering failed: {str(e)}")
        else:
            st.info("Upload DICOM volume files in sidebar")
    
    # Tab 6: Annotations
    with tab6:
        st.subheader("Image Annotation")
        
        if st.session_state.annotate_image and st.session_state.original_image:
            bg_image = st.session_state.original_image.convert("RGBA")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=bg_image,
                drawing_mode=drawing_mode,
                height=bg_image.height,
                width=bg_image.width,
                display_toolbar=True,
                key="canvas"
            )
            
            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data, caption="Annotated Image", use_container_width=True)
                if st.button("Save Annotation"):
                    annotated = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    st.session_state.annotated_image = annotated
                    st.success("Annotation saved")
        else:
            st.info("Enable annotation mode from sidebar")
    
    # Report Generation
    if st.session_state.generate_report:
        if all(k in st.session_state for k in ['original_image', 'cls', 'probs']):
            report = generate_diagnostic_report(
                st.session_state.original_image,
                st.session_state.adjusted_image,
                st.session_state.cls,
                st.session_state.probs,
                st.session_state.uncertainty,
                st.session_state.get('gradcam_image', st.session_state.original_image)
            )
            
            st.download_button(
                "Download Report",
                report,
                file_name="diagnostic_report.txt",
                mime="text/plain"
            )
            
            st.text_area("Report Preview", report, height=300)
            st.session_state.generate_report = False
        else:
            st.error("Complete prediction first")
    
    # Audit Log
    if st.session_state.view_audit_log:
        st.subheader("Audit Log")
        if st.session_state.audit_log:
            for entry in st.session_state.audit_log:
                st.text(entry)
        else:
            st.info("No audit entries yet")
        st.session_state.view_audit_log = False
    
    # EMR Export
    if st.session_state.export_to_emr:
        st.success("Results exported to EMR (simulated)")
        append_audit_log("Results exported to EMR")
        st.session_state.export_to_emr = False

if __name__ == "__main__":
    if STREAMLIT_SHARING:
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
    
    main()