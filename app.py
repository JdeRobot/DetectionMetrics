import streamlit as st
import os
import sys
import subprocess
from tabs.dataset_viewer import dataset_viewer_tab
from tabs.inference import inference_tab
from tabs.evaluator import evaluator_tab

def browse_folder():
    """
    Opens a native folder selection dialog and returns the selected folder path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).
    Returns None if cancelled or error.
    """
    try:
        if sys.platform.startswith("win"):
            script = (
                'Add-Type -AssemblyName System.windows.forms;'
                '$f=New-Object System.Windows.Forms.FolderBrowserDialog;'
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.SelectedPath}'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=30
            )
            folder = result.stdout.strip()
            return folder if folder else None
        elif sys.platform == "darwin":
            script = 'POSIX path of (choose folder with prompt "Select dataset folder:")'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            folder = result.stdout.strip()
            return folder if folder else None
        else:
            # Linux: try zenity, then kdialog
            for cmd in [
                ["zenity", "--file-selection", "--directory", "--title=Select dataset folder"],
                ["kdialog", "--getexistingdirectory", "--title", "Select dataset folder"]
            ]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    folder = result.stdout.strip()
                    if folder:
                        return folder
                except Exception:
                    continue
            return None
    except Exception:
        return None

st.set_page_config(page_title="DetectionMetrics", layout="wide")

# st.title("DetectionMetrics")

PAGES = {
    "Dataset Viewer": dataset_viewer_tab,
    "Inference": inference_tab,
    "Evaluator": evaluator_tab,
}

# Initialize commonly used session state keys
st.session_state.setdefault("dataset_path", "")
st.session_state.setdefault("dataset_type_selectbox", "Coco")
st.session_state.setdefault("split_selectbox", "val")
st.session_state.setdefault("config_option", "Manual Configuration")
st.session_state.setdefault("confidence_threshold", 0.5)
st.session_state.setdefault("nms_threshold", 0.5)
st.session_state.setdefault("max_detections", 100)
st.session_state.setdefault("device", "cpu")
st.session_state.setdefault("batch_size", 1)
st.session_state.setdefault("detection_model", None)
st.session_state.setdefault("detection_model_loaded", False)

# Sidebar: Dataset Inputs
with st.sidebar:
    with st.expander("Dataset Inputs", expanded=True):
        # First row: Type and Split
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Type",
                ["Coco", "Custom"],
                key="dataset_type_selectbox",
            )
        with col2:
            st.selectbox(
                "Split",
                ["train", "val"],
                key="split_selectbox",
            )
        
        # Second row: Path and Browse button
        col1, col2 = st.columns([3, 1])
        with col1:
            dataset_path_input = st.text_input(
                "Dataset Folder Path",
                value=st.session_state.get("dataset_path", ""),
                key="dataset_path_input",
            )
        with col2:
            st.markdown("<div style='margin-bottom: 1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("Browse", key="browse_button"):
                folder = browse_folder()
                if folder and os.path.isdir(folder):
                    st.session_state["dataset_path"] = folder
                    st.rerun()
                elif folder is not None:
                    st.warning("Selected path is not a valid folder.")
        
        if dataset_path_input != st.session_state.get("dataset_path", ""):
            st.session_state["dataset_path"] = dataset_path_input

    with st.expander("Model Inputs", expanded=False):
        st.file_uploader(
            "Model File (.pt, .onnx, .h5, .pb, .pth)",
            type=["pt", "onnx", "h5", "pb", "pth"],
            key="model_file",
            help="Upload your trained model file.",
        )
        st.file_uploader(
            "Ontology File (.json)",
            type=["json"],
            key="ontology_file",
            help="Upload a JSON file with class labels.",
        )
        st.radio(
            "Configuration Method:",
            ["Manual Configuration", "Upload Config File"],
            key="config_option",
            horizontal=True,
        )
        if st.session_state.get("config_option", "Manual Configuration") == "Upload Config File":
            st.file_uploader(
                "Configuration File (.json)",
                type=["json"],
                key="config_file",
                help="Upload a JSON configuration file.",
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("confidence_threshold", 0.5),
                    step=0.01,
                    key="confidence_threshold",
                    help="Minimum confidence score for detections",
                )
                st.slider(
                    "NMS Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("nms_threshold", 0.5),
                    step=0.01,
                    key="nms_threshold",
                    help="Non-maximum suppression threshold",
                )
                st.number_input(
                    "Max Detections/Image",
                    min_value=1,
                    max_value=1000,
                    value=st.session_state.get("max_detections", 100),
                    step=1,
                    key="max_detections",
                )
            with col2:
                st.selectbox(
                    "Device",
                    ["cpu", "gpu"],
                    index=0 if st.session_state.get("device", "cpu") == "cpu" else 1,
                    key="device",
                )
                st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=256,
                    value=st.session_state.get("batch_size", 1),
                    step=1,
                    key="batch_size",
                )

        # Load model action in sidebar
        from detectionmetrics.models.torch_detection import TorchImageDetectionModel
        import json, tempfile

        
        load_model_btn = st.button(
            "Load Model",
            type="primary",
            use_container_width=True,
            help="Load and save the model for use in the Inference tab",
            key="sidebar_load_model_btn",
        )

        if load_model_btn:
            model_file = st.session_state.get("model_file")
            ontology_file = st.session_state.get("ontology_file")
            config_option = st.session_state.get("config_option", "Manual Configuration")
            config_file = st.session_state.get("config_file") if config_option == "Upload Config File" else None

            # Prepare configuration
            config_data = None
            config_path = None
            try:
                if config_option == "Upload Config File":
                    if config_file is not None:
                        config_data = json.load(config_file)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
                            json.dump(config_data, tmp_cfg)
                            config_path = tmp_cfg.name
                    else:
                        st.error("Please upload a configuration file")
                else:
                    confidence_threshold = float(st.session_state.get('confidence_threshold', 0.5))
                    nms_threshold = float(st.session_state.get('nms_threshold', 0.5))
                    max_detections = int(st.session_state.get('max_detections', 100))
                    device = st.session_state.get('device', 'cpu')
                    batch_size = int(st.session_state.get('batch_size', 1))
                    config_data = {
                        "confidence_threshold": confidence_threshold,
                        "nms_threshold": nms_threshold,
                        "max_detections_per_image": max_detections,
                        "device": device,
                        "batch_size": batch_size,
                    }
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
                        json.dump(config_data, tmp_cfg)
                        config_path = tmp_cfg.name
            except Exception as e:
                st.error(f"Failed to prepare configuration: {e}")
                config_path = None

            if model_file is None:
                st.error("Please upload a model file")
            elif config_path is None:
                st.error("Please provide a valid model configuration")
            elif ontology_file is None:
                st.error("Please upload an ontology file")
            else:
                with st.spinner("Loading model..."):
                    # Persist ontology to temp file
                    try:
                        ontology_data = json.load(ontology_file)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_ont:
                            json.dump(ontology_data, tmp_ont)
                            ontology_path = tmp_ont.name
                    except Exception as e:
                        st.error(f"Failed to load ontology: {e}")
                        ontology_path = None

                    # Persist model to temp file
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt', mode='wb') as tmp_model:
                            tmp_model.write(model_file.read())
                            model_temp_path = tmp_model.name
                    except Exception as e:
                        st.error(f"Failed to save model file: {e}")
                        model_temp_path = None

                    if ontology_path and model_temp_path:
                        try:
                            model = TorchImageDetectionModel(
                                model=model_temp_path,
                                model_cfg=config_path,
                                ontology_fname=ontology_path,
                                device=st.session_state.get('device', 'cpu'),
                            )
                            st.session_state.detection_model = model
                            st.session_state.detection_model_loaded = True
                            st.success("Model loaded and saved for inference")
                        except Exception as e:
                            st.session_state.detection_model = None
                            st.session_state.detection_model_loaded = False
                            st.error(f"Failed to load model: {e}")

# Main content area with horizontal tabs
tab1, tab2, tab3 = st.tabs(["Dataset Viewer", "Inference", "Evaluator"])

with tab1:
    dataset_viewer_tab()
with tab2:
    inference_tab()
with tab3:
    evaluator_tab()