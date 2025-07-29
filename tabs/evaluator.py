import streamlit as st
import os
import tempfile
import json
import pandas as pd
from detectionmetrics.models.torch_detection import TorchImageDetectionModel
from detectionmetrics.datasets.coco import CocoDataset

def evaluator_tab():
    st.header("Evaluator")
    st.markdown("Evaluate your model on the loaded dataset using detection metrics.")
    
    # Check if we have the required objects from other tabs
    dataset_available = False
    model_available = False
    
    # Check for dataset from dataset viewer tab
    if 'dataset_path' in st.session_state and st.session_state['dataset_path']:
        dataset_path = st.session_state['dataset_path']
        dataset_type = st.session_state.get('dataset_type_selectbox', 'Coco')
        split = st.session_state.get('split_selectbox', 'val')
        
        # Try to load the dataset
        try:
            if dataset_type.lower() == "coco":
                img_dir = os.path.join(dataset_path, f"images/{split}2017")
                ann_file = os.path.join(dataset_path, "annotations", f"instances_{split}2017.json")
                
                if os.path.isdir(img_dir) and os.path.isfile(ann_file):
                    dataset_key = f"{dataset_path}_{split}"
                    if dataset_key not in st.session_state:
                        st.session_state[dataset_key] = CocoDataset(annotation_file=ann_file, image_dir=img_dir, split=split)
                        # Make filenames global - this is crucial for evaluation
                        st.session_state[dataset_key].make_fname_global()
                    dataset = st.session_state[dataset_key]
                    dataset_available = True
                    st.success(f"✅ Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples")
                else:
                    st.warning("⚠️ Dataset files not found. Please load a dataset in the Dataset Viewer tab.")
            else:
                st.warning("⚠️ Only COCO datasets are currently supported for evaluation.")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    else:
        st.warning("⚠️ No dataset loaded. Please load a dataset in the Dataset Viewer tab.")
    
    # Check for model from inference tab
    if 'detection_model' in st.session_state and st.session_state.detection_model is not None:
        model = st.session_state.detection_model
        model_available = True
        st.success("✅ Model loaded and ready for evaluation")
    else:
        st.warning("⚠️ No model loaded. Please load a model in the Inference tab.")
    
    # Evaluation configuration
    st.markdown("### Evaluation Configuration")
    
    save_predictions = st.checkbox(
        "Save Predictions",
        value=False,
        help="Save individual predictions and metrics per sample"
    )

    ontology_translation = st.file_uploader(
        "Ontology Translation (Optional)",
        type=["json"],
        help="JSON file for translating between dataset and model ontologies"
    )
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary", disabled=not (dataset_available and model_available)):
        if not dataset_available or not model_available:
            st.error("Please ensure both dataset and model are loaded before running evaluation.")
            return
        
        # Prepare evaluation
        with st.spinner("Running evaluation..."):
            try:
                # Validate dataset and model
                if len(dataset.dataset) == 0:
                    st.error("Dataset has no samples. Please check the dataset configuration.")
                    return
                
                if not hasattr(model, 'model_cfg') or model.model_cfg is None:
                    st.error("Model configuration is missing. Please reload the model in the Inference tab.")
                    return
                
                # Handle ontology translation if provided
                ontology_translation_path = None
                if ontology_translation is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_trans:
                        json.dump(json.load(ontology_translation), tmp_trans)
                        ontology_translation_path = tmp_trans.name
                
                # Prepare predictions output directory if needed
                predictions_outdir = None
                if save_predictions:
                    predictions_outdir = tempfile.mkdtemp(prefix="eval_predictions_")
                
                # Use model config as is (no confidence threshold override)
                eval_config = model.model_cfg.copy()
                
                # Debug information
                st.info(f"Dataset has {len(dataset.dataset)} samples")
                st.info(f"Model configuration: {eval_config}")
                
                # Create progress bar for evaluation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(processed, total):
                    """Progress callback for Streamlit UI"""
                    progress = processed / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {processed}/{total} images ({progress:.1%})")
                
                # Run evaluation with progress tracking
                results = model.eval(
                    dataset=dataset,
                    split=split,
                    ontology_translation=ontology_translation_path,
                    predictions_outdir=predictions_outdir,
                    results_per_sample=save_predictions,
                    progress_callback=progress_callback
                )
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                # Store results in session state
                st.session_state['evaluation_results'] = results
                st.session_state['evaluation_config'] = {
                    'split': split,
                    'predictions_saved': save_predictions
                }
                
                st.success("✅ Evaluation completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results (either from current evaluation or previous)
    if 'evaluation_results' in st.session_state:
        display_evaluation_results(st.session_state['evaluation_results'])



def display_evaluation_results(results):
    """Display evaluation results in a comprehensive format"""
    
    if results is None or results.empty:
        st.warning("No evaluation results to display.")
        return
    
    # Convert results to DataFrame if it's not already
    if not isinstance(results, pd.DataFrame):
        st.error("Results format not supported.")
        return
    
    # Display summary metrics
    st.markdown("#### Summary Metrics")
    
    # Get mean metrics - mean is a column
    if 'mean' in results.columns:
        mean_metrics = results['mean']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("mAP", f"{mean_metrics.get('AP', 0):.3f}")
        with col2:
            st.metric("Mean Precision", f"{mean_metrics.get('Precision', 0):.3f}")
        with col3:
            st.metric("Mean Recall", f"{mean_metrics.get('Recall', 0):.3f}")
        with col4:
            total_detections = mean_metrics.get('TP', 0) + mean_metrics.get('FP', 0)
            st.metric("Total Detections", f"{total_detections:.0f}")
    
    # Display per-class metrics
    st.markdown("#### Per-Class Metrics")
    
    # Filter out the 'mean' column for per-class display
    per_class_results = results.drop(columns=['mean']) if 'mean' in results.columns else results
    
    # Create a more readable display
    display_df = per_class_results.copy()
    
    # Round numeric columns for better display
    numeric_columns = display_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download results
    st.markdown("#### Download Results")
    
    # Convert to CSV for download
    csv = results.to_csv(index=True)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )
    
    # Show detailed statistics
    with st.expander("📊 Detailed Statistics"):
        st.markdown("**Results Shape:**")
        st.write(f"Rows: {results.shape[0]}, Columns: {results.shape[1]}")
        
        st.markdown("**Available Metrics:**")
        st.write(list(results.columns))
        
        st.markdown("**Class Names:**")
        st.write(list(results.index) if len(results.index) > 0 else "No classes found")
        
        st.markdown("**DataFrame Info:**")
        st.write("Index:", results.index.tolist())
        st.write("Columns:", results.columns.tolist())
        
        st.markdown("**Sample Data:**")
        st.dataframe(results.head(), use_container_width=True)
        
        if 'evaluation_config' in st.session_state:
            st.markdown("**Evaluation Configuration:**")
            config = st.session_state['evaluation_config']
            for key, value in config.items():
                st.write(f"- {key}: {value}") 