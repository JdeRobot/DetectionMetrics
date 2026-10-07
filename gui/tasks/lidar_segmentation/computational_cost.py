import streamlit as st
import pandas as pd

def render_lidar_segmentation_cost():
    st.header("Computational Cost Analysis")
    st.markdown("Analyze the computational cost of the loaded LiDAR Segmentation model.")

    if "lidar_model" not in st.session_state or st.session_state.lidar_model is None:
        st.warning("⚠️ Load a model from the sidebar to analyze its computational cost")
        return

    st.success("Model is loaded and ready for analysis.")

    col1, col2 = st.columns(2)
    with col1:
        num_points = st.number_input("Number of Points", min_value=1000, value=100000, step=1000, key="ls_cost_pts")
        has_intensity = st.checkbox("Has Intensity", value=False, key="ls_cost_intensity")
        st.markdown("**Point Cloud Range**")
        pr1, pr2, pr3 = st.columns(3)
        x_min = pr1.number_input("X Min", value=-50, key="ls_cost_xmin")
        y_min = pr2.number_input("Y Min", value=-50, key="ls_cost_ymin")
        z_min = pr3.number_input("Z Min", value=-5, key="ls_cost_zmin")
        x_max = pr1.number_input("X Max", value=50, key="ls_cost_xmax")
        y_max = pr2.number_input("Y Max", value=50, key="ls_cost_ymax")
        z_max = pr3.number_input("Z Max", value=5, key="ls_cost_zmax")
        
    with col2:
        runs = st.number_input("Inference Runs (for averaging)", min_value=1, value=30, step=5, key="ls_cost_r")
        warm_up_runs = st.number_input("Warm-up Runs", min_value=0, value=5, step=1, key="ls_cost_w_r")

    if st.button("Calculate Computational Cost", type="primary", key="ls_cost_btn"):
        with st.spinner("Calculating cost..."):
            try:
                pc_range = (int(x_min), int(y_min), int(z_min), int(x_max), int(y_max), int(z_max))
                cost_dict = st.session_state.lidar_model.get_computational_cost(
                    point_cloud_range=pc_range,
                    num_points=int(num_points),
                    has_intensity=bool(has_intensity),
                    runs=int(runs),
                    warm_up_runs=int(warm_up_runs)
                )

                df = pd.DataFrame([cost_dict])
                
                st.markdown("### Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Input Shape", f"{num_points} pts")
                m2.metric("Parameters (M)", f"{cost_dict.get('parameters', 0)/1e6:.2f}")
                m3.metric("Model Size (MB)", f"{cost_dict.get('model_size_mb', 0):.2f}")
                inf_time = cost_dict.get('inference_time_ms', 0)
                m4.metric("Latency (ms)", f"{inf_time:.2f}")
                if inf_time > 0:
                    m5.metric("FPS", f"{1000/inf_time:.1f}")
                else:
                    m5.metric("FPS", "N/A")

                with st.expander("Detailed Results (DataFrame)", expanded=True):
                    st.dataframe(df, width=800)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name="computational_cost.csv",
                    mime="text/csv",
                    key="ls_cost_dl"
                )
            except Exception as e:
                st.error(f"Error calculating computational cost: {e}")
