import streamlit as st
import os
import json
import tempfile
from detectionmetrics.models.torch_detection import TorchImageDetectionModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import supervision as sv

def draw_detections(image, predictions, label_map=None):
    """
    Draw color-coded bounding boxes and labels on the image using supervision.
    Args:
        image: PIL Image
        predictions: dict with 'boxes', 'labels', 'scores' (torch tensors)
        label_map: dict mapping label indices to class names (optional)
    Returns:
        np.ndarray with detections drawn (for st.image)
    """
    import numpy as np
    from supervision.draw.color import ColorPalette
    from supervision.detection.annotate import BoxAnnotator
    from supervision.detection.core import Detections

    img_np = np.array(image)
    boxes = predictions.get('boxes', torch.empty(0)).cpu().numpy()
    labels = predictions.get('labels', torch.empty(0)).cpu().numpy()
    scores = predictions.get('scores', torch.empty(0)).cpu().numpy()
    if label_map:
        class_names = [label_map.get(label, str(label)) for label in labels]
    else:
        class_names = [str(label) for label in labels]
    unique_class_names = list({name for name in class_names})
    palette = ColorPalette.default()
    class_name_to_color = {name: palette.by_idx(i) for i, name in enumerate(unique_class_names)}
    box_colors = [class_name_to_color[name] for name in class_names]
    detections = Detections(
        xyxy=boxes,
        class_id=labels.astype(int)
    )
    annotator = BoxAnnotator(color=palette, text_scale=0.7, text_thickness=1, text_padding=2)
    annotated_img = annotator.annotate(
        scene=img_np,
        detections=detections,
        labels=[f"{name}: {score:.2f}" for name, score in zip(class_names, scores)]
    )
    return annotated_img

def inference_tab():
    st.header("Model Inference")
    st.markdown("Upload your model and run inference on images with object detection.")
    
    # Initialize session state
    if 'detection_model' not in st.session_state:
        st.session_state.detection_model = None
    
    # File upload section with better visual organization
    with st.container():
        st.markdown("### Required Files")
        
        # Model file and Ontology file in the same row
        col_model, col_ont = st.columns(2)
        with col_model:
            model_file = st.file_uploader(
                "**Model File** (.pt, .onnx, .h5, .pb, .pth)",
                type=["pt", "onnx", "h5", "pb", "pth"],
                help="Upload your trained model file. The model file should be a PyTorch model (TorchScript or native PyTorch .pt/.pth file). Other formats are not supported for inference in this app.",
                key="model_file"
            )
        with col_ont:
            ontology_file = st.file_uploader(
                "**Ontology File** (.json)",
                type=["json"],
                help="Upload a JSON file containing class labels and their mappings",
                key="ontology_file"
            )
        
        # Configuration section
        st.markdown("### Configuration")
        config_option = st.radio(
            "Configuration Method:",
            ["Manual Configuration", "Upload Config File"],
            horizontal=True
        )
        
        config_data = None
        config_path = None
        
        if config_option == "Upload Config File":
            config_file = st.file_uploader(
                "**Configuration File** (.json)",
                type=["json"],
                help="Upload a JSON configuration file",
                key="config_file"
            )
            if config_file is not None:
                try:
                    config_data = json.load(config_file)
                    st.success("Configuration loaded successfully")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
                        json.dump(config_data, tmp_cfg)
                        config_path = tmp_cfg.name
                except Exception as e:
                    st.error(f"Failed to load configuration: {e}")
        else:
            # Manual configuration in an expander
            with st.expander("Manual Configuration Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="Minimum confidence score for detections"
                    )
                    nms_threshold = st.slider(
                        "NMS Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="Non-maximum suppression threshold"
                    )
                    max_detections = st.number_input(
                        "Max Detections/Image",
                        min_value=1,
                        max_value=1000,
                        value=100,
                        step=1,
                        help="Maximum detections per image"
                    )
                
                with col2:
                    device = st.selectbox(
                        "Device",
                        ["cpu", "gpu"],
                        help="Device to run inference on"
                    )
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=256,
                        value=1,
                        step=1,
                        help="Batch size for inference"
                    )
                
                config_data = {
                    "confidence_threshold": confidence_threshold,
                    "nms_threshold": nms_threshold,
                    "max_detections_per_image": max_detections,
                    "device": device,
                    "batch_size": batch_size
                }
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
                    json.dump(config_data, tmp_cfg)
                    config_path = tmp_cfg.name
            
            st.info("Manual configuration will be used")
        
        ontology_path = None
        if ontology_file is not None:
            try:
                ontology_data = json.load(ontology_file)
                st.success("Ontology loaded successfully")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_ont:
                    json.dump(ontology_data, tmp_ont)
                    ontology_path = tmp_ont.name
            except Exception as e:
                st.error(f"Failed to load ontology: {e}")
    
    # Status indicator
    model_status = st.empty()
    if st.session_state.detection_model is not None:
        model_status.success("Model loaded and ready for inference")
    else:
        model_status.info("Model not loaded")
    
    # Load button with validation
    load_col1, load_col2, load_col3 = st.columns([1, 2, 1])
    with load_col2:
        load_model = st.button(
            "Load Model",
            type="primary",
            use_container_width=True,
            help="Load the model with the provided configuration"
        )
    
    if load_model:
        with st.spinner("Loading model..."):
            if model_file is None:
                st.error("Please upload a model file")
            elif config_path is None:
                st.error("Please provide a valid model configuration")
            elif ontology_path is None:
                st.error("Please upload an ontology file")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt', mode='wb') as tmp_model:
                        tmp_model.write(model_file.read())
                        model_temp_path = tmp_model.name
                    
                    model = TorchImageDetectionModel(
                        model=model_temp_path,
                        model_cfg=config_path,
                        ontology_fname=ontology_path,
                        device="cpu"
                    )
                    st.session_state.detection_model = model
                    model_status.success("Model loaded and ready for inference")
                except Exception as e:
                    st.session_state.detection_model = None
                    st.error(f"Failed to load model: {e}")
                    model_status.info("Model not loaded")
    
    # Inference section
    st.markdown("### Run Inference")
    
    # Check if model is loaded
    if st.session_state.detection_model is None:
        st.warning("Please load a model first")
    else:
        st.success("Model is saved for inference")
        
        # Image upload and inference section
        image_file = st.file_uploader(
            "**Image File** (.jpg, .jpeg, .png, .bmp)",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to run inference on",
            key="image_file"
        )
        
        if image_file is not None:
            with st.spinner("Running inference..."):
                try:
                    image = Image.open(image_file).convert("RGB")
                    predictions = st.session_state.detection_model.inference(image)
                    label_map = getattr(st.session_state.detection_model, 'idx_to_class_name', None)
                    result_img = draw_detections(image.copy(), predictions, label_map)
                    
                    st.markdown("#### Detection Results")
                    st.image(result_img, caption="Detection Results", use_container_width=True)
                    
                    # Display detection statistics
                    if predictions.get('scores') is not None and len(predictions['scores']) > 0:
                        st.markdown("#### Detection Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Detections", len(predictions['scores']))
                        with col2:
                            avg_confidence = float(predictions['scores'].mean())
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        with col3:
                            max_confidence = float(predictions['scores'].max())
                            st.metric("Max Confidence", f"{max_confidence:.3f}")
                        
                        # Display and download detection results
                        st.markdown("#### Detection Results")
                        
                        # Convert predictions to JSON format
                        detection_results = []
                        boxes = predictions.get('boxes', torch.empty(0)).cpu().numpy()
                        labels = predictions.get('labels', torch.empty(0)).cpu().numpy()
                        scores = predictions.get('scores', torch.empty(0)).cpu().numpy()
                        
                        for i in range(len(scores)):
                            # Get class name if available
                            class_name = label_map.get(int(labels[i]), f"class_{labels[i]}") if label_map else f"class_{labels[i]}"
                            
                            detection_results.append({
                                "detection_id": i,
                                "class_id": int(labels[i]),
                                "class_name": class_name,
                                "confidence": float(scores[i]),
                                "bbox": {
                                    "x1": float(boxes[i][0]),
                                    "y1": float(boxes[i][1]),
                                    "x2": float(boxes[i][2]),
                                    "y2": float(boxes[i][3])
                                },
                                "bbox_xyxy": boxes[i].tolist()
                            })
                        
                        # Display JSON in expandable section
                        with st.expander(" View Detection Results (JSON)", expanded=False):
                            st.json(detection_results)
                        
                        # Download JSON file
                        json_str = json.dumps(detection_results, indent=2)
                        st.download_button(
                            label="Download Detection Results as JSON",
                            data=json_str,
                            file_name="detection_results.json",
                            mime="application/json",
                            help="Download the detection results as a JSON file"
                        )
                    else:
                        st.info("No detections found in the image.")
                except Exception as e:
                    st.error(f"Failed to run inference: {e}") 