import streamlit as st
from tabs.dataset_viewer import dataset_viewer_tab
from tabs.inference import inference_tab
from tabs.evaluator import evaluator_tab

st.set_page_config(page_title="DetectionMetrics", layout="wide")

# st.title("DetectionMetrics")

PAGES = {
    "Dataset Viewer": dataset_viewer_tab,
    "Inference": inference_tab,
    "Evaluator": evaluator_tab
}

page = st.sidebar.radio("DetectionMetrics", list(PAGES.keys()))

PAGES[page]() 