import streamlit as st
from gui.tasks.image_detection.computational_cost import render_image_detection_cost
from gui.tasks.image_segmentation.computational_cost import render_image_segmentation_cost
from gui.tasks.lidar_segmentation.computational_cost import render_lidar_segmentation_cost


def computational_cost_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        render_image_detection_cost()
        return

    if task == "Image Segmentation":
        render_image_segmentation_cost()
        return

    if task == "Lidar Segmentation":
        render_lidar_segmentation_cost()
        return

    st.error(f"Unsupported task for computational cost: {task}")
