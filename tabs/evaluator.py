import streamlit as st
from tabs.tasks.image_detection.evaluator import render_image_detection_evaluator
from tabs.tasks.image_segmentation.evaluator import render_image_segmentation_evaluator
from tabs.tasks.lidar_segmentation.evaluator import render_lidar_segmentation_evaluator


def evaluator_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        render_image_detection_evaluator()
        return

    if task == "Image Segmentation":
        render_image_segmentation_evaluator()
        return

    if task == "Lidar Segmentation":
        render_lidar_segmentation_evaluator()
        return

    st.error(f"Unsupported task for evaluator: {task}")
