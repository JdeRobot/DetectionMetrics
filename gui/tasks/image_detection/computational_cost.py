import streamlit as st
import pandas as pd

def render_image_detection_cost():
    st.header("Computational Cost Analysis")
    st.markdown("Analyze the computational cost of the loaded Image Detection model.")

    if "detection_model" not in st.session_state or st.session_state.detection_model is None:
        st.warning("⚠️ Load a model from the sidebar to analyze its computational cost")
        return

    st.success("Model is loaded and ready for analysis.")

    col1, col2 = st.columns(2)
    with col1:
        img_height = st.number_input("Input Height", min_value=32, value=640, step=32, key="id_cost_h")
        img_width = st.number_input("Input Width", min_value=32, value=640, step=32, key="id_cost_w")
    with col2:
        runs = st.number_input("Inference Runs (for averaging)", min_value=1, value=30, step=5, key="id_cost_r")
        warm_up_runs = st.number_input("Warm-up Runs", min_value=0, value=5, step=1, key="id_cost_w_r")

    if st.button("Calculate Computational Cost", type="primary", key="id_cost_btn"):
        with st.spinner("Calculating cost..."):
            try:
                img_size = (int(img_height), int(img_width))
                cost_dict = st.session_state.detection_model.get_computational_cost(
                    image_size=img_size,
                    runs=int(runs),
                    warm_up_runs=int(warm_up_runs)
                )

                df = pd.DataFrame([cost_dict])
                
                st.markdown("### Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Input Shape", f"{img_height}x{img_width}")
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
                    key="id_cost_dl"
                )
            except Exception as e:
                st.error(f"Error calculating computational cost: {e}")
