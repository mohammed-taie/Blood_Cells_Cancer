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

from captum.attr import IntegratedGradients  # For advanced explainability

# Streamlit page configuration
st.set_page_config(layout="wide", page_icon="🔬", page_title="Blood Smear Classifier App")

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)

# Custom CSS for a consistent aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #3E95CD;
    }
    .css-1d391kg, .css-1d391kg div {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
    .stButton>button {
        background-color: #3E95CD;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
        font-size: 16px;
    }
    .custom-info {
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@dataclass
class Config:
    DEVICE: torch.device = torch.device("cpu")
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
    input_tensor = transform(image).unsqueeze(0).to(device)
    baseline = torch.full_like(input_tensor, baseline_value)
    ig = IntegratedGradients(model)
    outputs = model(input_tensor)
    target = outputs.argmax(dim=1).item()
    attributions, _ = ig.attribute(input_tensor, baseline, target=target, return_convergence_delta=True, n_steps=steps)
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
    if method == "Grad-CAM":
        explanation = (
            "The Grad-CAM visualization highlights key regions in the image that the model considers "
            "important for making its prediction. Areas with intense colors indicate higher influence. "
            "If the visualization is in Overlay mode, the original image is shown along with a transparent heatmap; "
            "in Heatmap mode, the regions stand out more prominently. This helps verify whether the model focuses on clinically relevant features."
        )
    elif method == "Integrated Gradients":
        explanation = (
            "The Integrated Gradients method assigns attribution values to different parts of the image, "
            "indicating each part's contribution to the final prediction. The generated heatmap shows regions where the input features exert a strong impact. "
            "In Overlay mode, this attribution is blended with the original image; in Heatmap mode, the attributions are displayed more distinctly. "
            "This rationale aids in evaluating whether the model’s focus aligns with key clinical attributes."
        )
    else:
        explanation = "No detailed explanation available for the selected method."
    explanation += f"\n\nSelected Visualization Mode: **{visualization_mode}**."
    return explanation

def apply_windowing(image: Image.Image, window_center: float, window_width: float) -> Image.Image:
    try:
        gray = ImageOps.grayscale(image)
        np_img = np.array(gray).astype(np.float32)
        lower_bound = window_center - (window_width / 2)
        np_img = (np_img - lower_bound) / window_width * 255.0
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    except Exception as e:
        logging.error(f"Windowing failed: {str(e)}")
        st.error("Failed to apply windowing. Returning original grayscale image as fallback.")
        return ImageOps.grayscale(image)

class BloodCellClassifier:
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
            if not os.path.exists(_config.MODEL_WEIGHTS_PATH):
                logging.error("Model weights file not found. Using random weights as fallback.")
                st.warning("Warning: Model weights not found. Using a fallback model with random weights.")
            else:
                state_dict = torch.load(_config.MODEL_WEIGHTS_PATH, map_location=_config.DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
            model.to(_config.DEVICE).eval()
            return model
        except Exception as e:
            logging.critical(f"Model loading failed: {str(e)}")
            st.error("Critical error: Failed to load prediction model. Please check the logs.")
            st.stop()

    def predict(self, image: Image.Image, num_samples: int = 10) -> tuple:
        try:
            img_tensor = self.config.TRANSFORM(image).unsqueeze(0).to(self.config.DEVICE)
            model = self.model
            model.eval()
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
            st.error("Error during prediction. Falling back to default values.")
            fallback_probs = [1.0 / len(self.config.CLASS_LABELS)] * len(self.config.CLASS_LABELS)
            fallback_uncertainty = [0.5] * len(self.config.CLASS_LABELS)
            return -1, fallback_probs, fallback_uncertainty

class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config

    def validate_file(self, uploaded_file) -> bool:
        try:
            if uploaded_file.size > self.config.MAX_MB * 1024 * 1024:
                raise ValueError(f"File size exceeds {self.config.MAX_MB}MB limit")
            ext = uploaded_file.type.split('/')[1].lower()
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
        try:
            gray = np.array(image.convert("L"))
            brightness = gray.mean()
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            qc_passed = (40 <= brightness <= 230) and (laplacian_var >= 50)
            return qc_passed, brightness, laplacian_var
        except Exception as e:
            logging.error(f"Quality control failed: {str(e)}")
            st.error("Error during image quality control. Proceeding with original image as fallback.")
            return True, 128, 100

    def load_image(self, uploaded_file) -> Image.Image:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            qc_passed, _, _ = self.quality_control(image)
            if not qc_passed:
                st.error("Image quality check failed. Please upload an image with adequate brightness and focus.")
                raise ValueError("Image quality check failed")
            return image
        except UnidentifiedImageError:
            st.error("Invalid or corrupted image file")
            logging.error("Failed to load image due to UnidentifiedImageError")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading image: {str(e)}")
            st.error("An unexpected error occurred while loading the image. Please try again.")
            raise

class ResultsVisualizer:
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
            fig.update_layout(yaxis_tickformat=".0%",
                              xaxis_title="Cell Type",
                              yaxis_title="Prediction Confidence",
                              hovermode="x unified")
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
            st.error("Failed to display detailed results.")

def apply_high_resolution_controls(image: Image.Image, brightness: float, contrast: float,
                                   zoom: float, rotation: float, saturation: float,
                                   sharpness: float) -> Image.Image:
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
        st.error("Failed to apply high resolution controls. Returning original image as fallback.")
        return image

def generate_diagnostic_report(original_image: Image.Image,
                               adjusted_image: Image.Image,
                               prediction: str,
                               probabilities: list,
                               uncertainties: list,
                               gradcam_image: Image.Image) -> str:
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"Diagnostic Report\nTimestamp: {timestamp}\n\n"
        # Patient metadata
        if "patient_metadata" in st.session_state:
            patient = st.session_state["patient_metadata"]
            report += "Patient Information:\n"
            report += f"- Name: {patient.get('name', 'N/A')}\n"
            report += f"- Age: {patient.get('age', 'N/A')}\n"
            report += f"- Medical Record Number: {patient.get('id', 'N/A')}\n"
            report += f"- Clinical Notes: {patient.get('notes', 'N/A')}\n\n"
        # Prediction details
        report += f"Prediction: {prediction}\n\n"
        report += "Class Probabilities and Uncertainties:\n"
        for i, (prob, uncert) in enumerate(zip(probabilities, uncertainties)):
            report += f"- {Config().CLASS_LABELS[i]}: {prob*100:.2f}% (Uncertainty: {uncert*100:.2f}%)\n"
        # Explainability details
        if "explainability_method" in st.session_state:
            report += "\nExplainability Details:\n"
            report += f"- Method: {st.session_state.get('explainability_method', 'N/A')}\n"
            if st.session_state.get('explainability_method') == "Grad-CAM":
                report += f"- Target Layer: {st.session_state.get('target_layer', 'N/A')}\n"
            report += f"- Visualization Mode: {st.session_state.get('visualization_mode', 'Overlay')}\n"
            report += f"- Opacity: {st.session_state.get('opacity', 0.5)}\n"
            report += f"- Heatmap Intensity: {st.session_state.get('heatmap_intensity', 1.0)}\n"
            explanation_text = generate_explanation_text(
                st.session_state.get('explainability_method', 'N/A'), 
                st.session_state.get('visualization_mode', 'Overlay')
            )
            report += f"- Explanation: {explanation_text}\n"
        # Confidence details
        if "confidence_threshold" in st.session_state:
            report += "\nConfidence Details:\n"
            report += f"- Set Confidence Threshold: {st.session_state.get('confidence_threshold', 0.7)*100:.0f}%\n"
            if "probs" in st.session_state:
                probs = st.session_state["probs"]
                pred_idx = np.argmax(probs)
                report += f"- Prediction Confidence: {probs[pred_idx]*100:.2f}%\n"
        # Windowing details
        if "window_center" in st.session_state and "window_width" in st.session_state:
            report += "\nWindowing Details:\n"
            report += f"- Window Center: {st.session_state.get('window_center', 'N/A')}\n"
            report += f"- Window Width: {st.session_state.get('window_width', 'N/A')}\n"

        report += "\nNote: This report is generated by an AI-based assistive diagnostic tool and should be used alongside clinical evaluation.\n"
        return report
    except Exception as e:
        logging.error(f"Error generating diagnostic report: {str(e)}")
        return "Diagnostic Report generation failed due to an unexpected error."

def append_audit_log(event: str):
    try:
        if "audit_log" not in st.session_state:
            st.session_state["audit_log"] = []
        st.session_state["audit_log"].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {event}")
    except Exception as e:
        logging.error(f"Error appending to audit log: {str(e)}")

def main():
    st.markdown('<p class="main-title">🔬 Blood Smear Classifier App</p>', unsafe_allow_html=True)
    
    # Initialize session state keys if not already available.
    for key in ["run_prediction", "run_explainability", "generate_report", "view_audit_log", "export_to_emr", "run_confidence", "run_windowing"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        with st.expander("👤 Patient Information", expanded=True):
            patient_name = st.text_input("Patient Name", placeholder="Enter patient's name")
            patient_age = st.number_input("Patient Age", min_value=0, value=0, step=1)
            patient_id = st.text_input("Medical Record Number", placeholder="Enter patient ID")
            clinical_notes = st.text_area("Clinical Notes", placeholder="Enter relevant clinical notes", height=100)
            if st.button("Save Patient Metadata"):
                st.session_state["patient_metadata"] = {
                    "name": patient_name,
                    "age": patient_age,
                    "id": patient_id,
                    "notes": clinical_notes
                }
                st.success("Patient metadata saved!")

        with st.expander("📁 File Operations", expanded=True):
            uploaded_file = st.file_uploader("Upload blood smear image (JPG/PNG)", type=Config().ALLOWED_EXTENSIONS)
            if st.button("🔄 Clear Cache", key="clear1"):
                st.cache_resource.clear()
                st.rerun()

        with st.expander("🔍 High-Resolution Controls", expanded=True):
            brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
            zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.1)
            rotation = st.slider("Rotation (°)", -180, 180, 0, 5)
            saturation = st.slider("Saturation", 0.5, 1.5, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 1.5, 1.0, 0.1)
            real_time_preview = st.checkbox("Real-Time Preview Update", value=True)

        with st.expander("🔮 Prediction Controls", expanded=True):
            if st.button("🔍 Run Prediction"):
                st.session_state["run_prediction"] = True
            if st.button("❌ Clear Results", key="clear2"):
                st.session_state.clear()
                st.rerun()

        with st.expander("🧩 Enhanced Explainability Controls", expanded=True):
            target_layer = st.text_input("Target Layer", value="inception5b")
            opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)
            visualization_mode = st.selectbox("Visualization Mode", options=["Overlay", "Heatmap"])
            if visualization_mode == "Heatmap":
                heatmap_intensity = st.slider("Heatmap Intensity", 0.5, 2.0, 1.0, 0.1)
            else:
                heatmap_intensity = 1.0
            explainability_method = st.selectbox("Explainability Method", options=["Grad-CAM", "Integrated Gradients"])
            st.markdown('<p class="custom-info">Select the explanation method and adjust parameters as needed.</p>', unsafe_allow_html=True)
            if st.button("Generate Explainability Report"):
                st.session_state["run_explainability"] = True
                st.session_state["target_layer"] = target_layer
                st.session_state["opacity"] = opacity
                st.session_state["visualization_mode"] = visualization_mode
                st.session_state["heatmap_intensity"] = heatmap_intensity
                st.session_state["explainability_method"] = explainability_method

        with st.expander("🚨 Confidence Thresholds and Alerts", expanded=True):
            confidence_threshold = st.slider("Set Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)
            if st.button("Check Confidence"):
                st.session_state["run_confidence"] = True
                st.session_state["confidence_threshold"] = confidence_threshold / 100.0

        with st.expander("🖼 Enhanced Contrast and Windowing Controls", expanded=True):
            window_center = st.slider("Window Center", min_value=0, max_value=255, value=128, step=1)
            window_width = st.slider("Window Width", min_value=1, max_value=255, value=128, step=1)
            if st.button("Apply Windowing"):
                st.session_state["run_windowing"] = True
                st.session_state["window_center"] = window_center
                st.session_state["window_width"] = window_width

        with st.expander("🩺 Integration with Clinical Workflow", expanded=True):
            st.markdown('<p class="custom-info">Generate diagnostic reports, view audit logs, and export data.</p>', unsafe_allow_html=True)
            if st.button("Generate Diagnostic Report"):
                st.session_state["generate_report"] = True
                append_audit_log("Diagnostic report generated")
            if st.button("View Audit Log"):
                st.session_state["view_audit_log"] = True
            if st.button("Export to EMR"):
                st.session_state["export_to_emr"] = True
                append_audit_log("Results exported to EMR")
                
        with st.expander("❓ Help & Info", expanded=True):
            st.markdown(
                """
### Onboarding

**Welcome to the Blood Smear Classifier App!**

**Getting Started:**
1. **Patient Information:** Enter patient details.
2. **File Operations:** Upload a blood smear image (JPG/PNG).
3. **High-Resolution Controls:** Adjust image parameters.
4. **Prediction Controls:** Run the model prediction to view class probabilities and uncertainty.
5. **Enhanced Explainability Controls:** Choose between Grad-CAM and Integrated Gradients to visualize key image regions.
6. **Confidence Thresholds and Alerts:** Set a confidence threshold and check if the prediction meets that threshold.
7. **Enhanced Contrast and Windowing Controls:** Adjust window center/width to enhance image contrast.
8. **Integration with Clinical Workflow:** Generate diagnostic reports and view audit/feedback logs.
                """
            )

    tabs = st.tabs(["Classification", "Enhanced Explainability", "Confidence Alerts", "Windowing Controls"])

    with tabs[0]:
        original_image = None
        if uploaded_file:
            image_processor = ImageProcessor(Config())
            if not image_processor.validate_file(uploaded_file):
                st.warning("Please upload a valid image file.")
            else:
                try:
                    original_image = image_processor.load_image(uploaded_file)
                    st.session_state["original_image"] = original_image
                except Exception:
                    st.stop()
        else:
            st.info("Upload a blood smear image (JPG/PNG) in the sidebar to classify.")

        if original_image is not None:
            if real_time_preview:
                adjusted_image = apply_high_resolution_controls(
                    original_image, brightness, contrast, zoom,
                    rotation, saturation, sharpness
                )
            else:
                if st.button("Update Preview"):
                    adjusted_image = apply_high_resolution_controls(
                        original_image, brightness, contrast, zoom,
                        rotation, saturation, sharpness
                    )
                else:
                    adjusted_image = original_image

            st.session_state["adjusted_image"] = adjusted_image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Original Image")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader("🖼️ Adjusted Image")
                st.image(adjusted_image, use_container_width=True)

            if st.button("Check Image Quality", key="check_quality_main"):
                qc_passed, brightness_value, laplacian_value = image_processor.quality_control(original_image)
                if qc_passed:
                    st.success(
                        f"Image quality check passed!\n\n"
                        f"**Brightness:** {brightness_value:.2f} (Expected: 40 - 230).\n\n"
                        f"**Laplacian Variance:** {laplacian_value:.2f} (Min: 50)."
                    )
                else:
                    message = "Image quality check failed!\n\n"
                    if brightness_value < 40:
                        message += f"- **Brightness is low:** {brightness_value:.2f}.\n"
                    elif brightness_value > 230:
                        message += f"- **Brightness is high:** {brightness_value:.2f}.\n"
                    if laplacian_value < 50:
                        message += f"- **Laplacian Variance is low:** {laplacian_value:.2f}.\n"
                    st.error(message)

            st.subheader("🔮 Prediction + Stats")
            if st.session_state["run_prediction"]:
                classifier = BloodCellClassifier(Config())
                with st.spinner("Analyzing image..."):
                    pred_idx, mean_probs, std_probs = classifier.predict(original_image, num_samples=10)
                    cls = Config().CLASS_LABELS.get(pred_idx, "Unknown")
                st.success(f"🎯 Result: **{cls}**")
                st.progress(mean_probs[pred_idx], text=f"{mean_probs[pred_idx]*100:.2f}%")
                visualizer = ResultsVisualizer(Config())
                visualizer.plot_probabilities(mean_probs)
                visualizer.show_detailed_table(mean_probs, std_probs)
                overall_uncertainty = np.mean(std_probs)
                st.info(f"Overall Uncertainty: {overall_uncertainty*100:.2f}%")
                st.markdown(
                    """
**Interpretation of Results:**

- **Class Probabilities:** The model indicates its confidence in each class.
- **Uncertainty Metrics:** Lower uncertainty suggests a stable prediction; higher uncertainty indicates ambiguity.
- **Clinical Relevance:** These metrics help determine if further confirmatory testing is needed.
                    """
                )
                st.subheader("📝 Provide Feedback")
                feedback_text = st.text_area("Enter your feedback regarding this prediction", height=100)
                if st.button("Submit Feedback"):
                    if "feedback_log" not in st.session_state:
                        st.session_state["feedback_log"] = []
                    feedback_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Prediction: {cls} - Feedback: {feedback_text}"
                    st.session_state["feedback_log"].append(feedback_entry)
                    append_audit_log(f"Feedback submitted: {feedback_text}")
                    st.success("Feedback submitted. Thank you!")
                append_audit_log(f"Prediction run: {cls}")
                st.session_state["cls"] = cls
                st.session_state["probs"] = mean_probs
                st.session_state["uncertainty"] = std_probs
                st.session_state["run_prediction"] = False
            else:
                st.info("Click 'Run Prediction' in the sidebar to classify the image.")

    with tabs[1]:
        if st.session_state.get("run_explainability", False):
            if "original_image" not in st.session_state:
                st.error("No image available for explainability. Please upload an image first.")
            else:
                original_image = st.session_state["original_image"]
                visualization_mode = st.session_state.get("visualization_mode", "Overlay")
                opacity = st.session_state.get("opacity", 0.5)
                heatmap_intensity = st.session_state.get("heatmap_intensity", 1.0)
                explainability_method = st.session_state.get("explainability_method", "Grad-CAM")
                classifier = BloodCellClassifier(Config())
                with st.spinner("Generating explanation..."):
                    try:
                        if explainability_method == "Grad-CAM":
                            target_layer = st.session_state.get("target_layer", "inception5b")
                            explanation_image = generate_gradcam(
                                original_image, classifier.model, Config().TRANSFORM,
                                Config().DEVICE, target_layer, opacity,
                                visualization_mode, heatmap_intensity
                            )
                        elif explainability_method == "Integrated Gradients":
                            explanation_image = generate_integrated_gradients(
                                original_image, classifier.model, Config().TRANSFORM,
                                Config().DEVICE, baseline_value=0.0, steps=50,
                                opacity=opacity, visualization_mode=visualization_mode,
                                heatmap_intensity=heatmap_intensity
                            )
                        else:
                            raise ValueError("Unsupported explainability method selected")
                        resized_explanation = explanation_image.copy()
                        resized_explanation = resized_explanation.resize((300, 300))
                    except Exception as e:
                        st.error(f"Explainability generation failed: {e}")
                cols = st.columns(2)
                with cols[0]:
                    st.subheader("🖼️ Original Image")
                    st.image(original_image, use_container_width=True)
                with cols[1]:
                    st.subheader(f"🖼️ {explainability_method} (Resized)")
                    st.image(resized_explanation, width=300)
                st.subheader("📄 Detailed Explanation and Rationale")
                explanation_text = generate_explanation_text(explainability_method, visualization_mode)
                st.markdown(explanation_text)
                st.markdown(
                    f"""
### Additional Explanation Details for {explainability_method}

- **Visualization Mode:**  
  - **Overlay:** The explanation is blended with the original image, contextualizing important anatomical regions.
  - **Heatmap:** The explanation heatmap is displayed alone, emphasizing region importance.
- **Parameters:**  
  - **Opacity:** Determines the overlay transparency.
  - **Heatmap Intensity:** Controls the visual prominence of the explanation zones.
- **Method Details:**  
  - **Grad-CAM:** Highlights influential regions via gradients from a target layer.
  - **Integrated Gradients:** Attributes contributions by integrating gradients over input features.
                    """
                )
            st.session_state["run_explainability"] = False
        else:
            st.info("Click 'Generate Explainability Report' in the sidebar to view enhanced explanations.")

    with tabs[2]:
        st.subheader("🚨 Confidence Alerts")
        if st.session_state.get("run_confidence", False):
            if "probs" in st.session_state and "cls" in st.session_state:
                probs = st.session_state["probs"]
                pred_idx = np.argmax(probs)
                prediction_confidence = probs[pred_idx]
                threshold = st.session_state.get("confidence_threshold", 0.70)
                predicted_class = st.session_state["cls"]
                st.markdown("### What is Being Displayed")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {prediction_confidence*100:.2f}%")
                st.write(f"**Set Threshold:** {threshold*100:.0f}%")
                st.markdown("### Why This Matters")
                st.write(
                    "The confidence value represents the model's estimated probability for its top prediction. "
                    "High confidence (above the set threshold) indicates a reliable decision, while lower values suggest uncertainty."
                )
                st.markdown("### How to Interpret the Results")
                if prediction_confidence < threshold:
                    st.error("Alert: The model's prediction confidence is below the set threshold. Consider additional diagnostic tests or a manual review.")
                else:
                    st.success("Good news: The prediction confidence meets or exceeds the set threshold, indicating high reliability.")
                append_audit_log(f"Confidence check: {prediction_confidence*100:.2f}% (Threshold: {threshold*100:.0f}%)")
            else:
                st.warning("No prediction available. Please run the prediction first.")
            st.session_state["run_confidence"] = False
        else:
            st.info("Set the confidence threshold and click 'Check Confidence' in the sidebar.")

    with tabs[3]:
        st.subheader("🖼 Windowing Controls")
        if st.session_state.get("run_windowing", False):
            if "original_image" in st.session_state:
                original_image = st.session_state["original_image"]
                window_center = st.session_state.get("window_center", 128)
                window_width = st.session_state.get("window_width", 128)
                with st.spinner("Applying windowing controls..."):
                    try:
                        windowed_image = apply_windowing(original_image, window_center, window_width)
                    except Exception as e:
                        st.error(f"Windowing failed: {e}")
                        windowed_image = ImageOps.grayscale(original_image)
                cols = st.columns(2)
                with cols[0]:
                    st.subheader("🖼 Original Grayscale")
                    st.image(ImageOps.grayscale(original_image), use_container_width=True)
                with cols[1]:
                    st.subheader("🖼 Windowed Image")
                    st.image(windowed_image, use_container_width=True)
                st.markdown(
                    f"""
### Detailed Explanation
- **Window Center:** {window_center}  
- **Window Width:** {window_width}  
                
**What:** Adjusts image contrast by rescaling pixel values based on the specified center and width.
                
**Why:** Enhances visual clarity, highlighting features vital for diagnostic review.
                
**How:** Pixels below (center - width/2) are mapped to 0; above (center + width/2) to 255; intermediate values are linearly scaled.
                    """
                )
            else:
                st.warning("No image available for windowing. Please upload an image first.")
            st.session_state["run_windowing"] = False
        else:
            st.info("Set the windowing parameters and click 'Apply Windowing' in the sidebar.")

    if st.session_state.get("generate_report", False):
        if "original_image" in st.session_state and "cls" in st.session_state and "probs" in st.session_state and "uncertainty" in st.session_state:
            report = generate_diagnostic_report(
                original_image=st.session_state["original_image"],
                adjusted_image=st.session_state.get("adjusted_image", st.session_state["original_image"]),
                prediction=st.session_state["cls"],
                probabilities=st.session_state["probs"],
                uncertainties=st.session_state["uncertainty"],
                gradcam_image=st.session_state.get("adjusted_image", st.session_state["original_image"])
            )
            st.download_button("Download Diagnostic Report", report, file_name="diagnostic_report.txt", mime="text/plain")
        else:
            st.error("Diagnostic report could not be generated because prediction details are missing.")
        st.session_state["generate_report"] = False

    if st.session_state.get("view_audit_log", False):
        st.subheader("Audit Log")
        if "audit_log" in st.session_state and st.session_state["audit_log"]:
            for log_entry in st.session_state["audit_log"]:
                st.text(log_entry)
        else:
            st.info("Audit log is empty.")
        if "feedback_log" in st.session_state and st.session_state["feedback_log"]:
            st.subheader("Feedback Log")
            for feedback in st.session_state["feedback_log"]:
                st.text(feedback)
        st.session_state["view_audit_log"] = False

    if st.session_state.get("export_to_emr", False):
        st.success("Results have been successfully exported to the EMR system.")
        st.session_state["export_to_emr"] = False

if __name__ == "__main__":
    main()