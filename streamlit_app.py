# =========================
# IMPORT LIBRARY
# =========================

import streamlit as st
import time

from model_loader import models
from predict import predict_all_models
from preprocessing import preprocess_image, preprocess_for_model
from gradcam import (
    get_gradcam_config,
    make_gradcam_heatmap,
    overlay_gradcam
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MRI Classification System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:0rem;
}

.header-title{
    text-align:center;
    font-size:24px;
    font-weight:700;
    color:white;
    margin-bottom:0px;
}

.header-subtitle{
    text-align:center;
    font-size:13px;
    color:#B0B0B0;
    margin-top:0px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<style>

.block-container{
    padding-top:3rem;
}

.header-title{
    text-align:center;
    font-size:24px;
    font-weight:600;
    color:white;
    margin-bottom:4px;
}

.header-subtitle{
    text-align:center;
    font-size:13px;
    color:#B0B0B0;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-title">
🧠 Brain Tumor MRI Classification System Using Deep Learning and Grad-CAM
</div>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "results" not in st.session_state:
    st.session_state.results = None

# =========================
# LAYOUT
# =========================

left_col, right_col = st.columns([1, 2])

# ==================================================
# LEFT PANEL
# ==================================================

with left_col:

    st.subheader("Upload MRI Image")

    uploaded_file = st.file_uploader(
        "Choose MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    with st.expander("⚠️ System Limitations"):
        st.markdown("""
        1. Sistem hanya dapat mengklasifikasikan 4 kelas: Glioma, Meningioma, Pituitary, dan No Tumor.  
        2. Sistem ini dibuat untuk tujuan penelitian, bukan untuk penggunaan klinis.  
        3. Hasil tidak boleh digunakan sebagai diagnosis medis resmi.
        """)

    analyze_btn = st.button(
        "🔍 Analyze MRI",
        use_container_width=True
    )

    # =========================
    # MRI PREVIEW
    # =========================

    if uploaded_file is not None:

        image_array, original_image = preprocess_image(
            uploaded_file
        )

        st.caption("MRI Image Preview")

        st.image(
            original_image,
            width=250
        )

# ==================================================
# ANALYZE
# ==================================================

if analyze_btn and uploaded_file is not None:

    with st.spinner("Analyzing MRI..."):

        time.sleep(1)

        results = predict_all_models(
            models,
            image_array,
            preprocess_for_model
        )

        dense = next(
            x for x in results
            if x["Model"] == "DenseNet121 FT"
        )

        resnet = next(
            x for x in results
            if x["Model"] == "ResNet50 FT"
        )

        efficient = next(
            x for x in results
            if x["Model"] == "EfficientNet FT"
        )

        best_model = max(
            [dense, resnet, efficient],
            key=lambda x: float(x["Confidence"])
        )

        # =========================
        # GRAD-CAM
        # =========================

        best_model_name = best_model["Model"]

        selected_model = models[
            best_model_name
        ]

        config = get_gradcam_config(
            best_model_name
        )

        processed_image = preprocess_for_model(
            image_array.copy(),
            best_model_name
        )

        heatmap = make_gradcam_heatmap(
            processed_image,
            selected_model,
            config["backbone_name"],
            config["last_conv_layer_name"]
        )

        heatmap_color, overlay = overlay_gradcam(
            original_image,
            heatmap
        )

        st.session_state.results = {
            "dense": dense,
            "resnet": resnet,
            "efficient": efficient,
            "best_model": best_model,
            "original": original_image,
            "heatmap": heatmap_color,
            "overlay": overlay
        }

        st.session_state.result_ready = True

# ==================================================
# RIGHT PANEL
# ==================================================

with right_col:

    if st.session_state.result_ready:

        data = st.session_state.results

        dense = data["dense"]
        resnet = data["resnet"]
        efficient = data["efficient"]
        best_model = data["best_model"]

        # =========================
        # MODEL PREDICTIONS
        # =========================

        st.subheader("Model Predictions")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "DenseNet121 FT",
                dense["Prediction"],
                f"{float(dense['Confidence']):.2f}%"
            )

        with c2:
            st.metric(
                "ResNet50 FT",
                resnet["Prediction"],
                f"{float(resnet['Confidence']):.2f}%"
            )

        with c3:
            st.metric(
                "EfficientNet FT",
                efficient["Prediction"],
                f"{float(efficient['Confidence']):.2f}%"
            )

        # =========================
        # FINAL RESULT
        # =========================

        st.subheader("Best Model Result")

        st.success(
            f"""
Prediction : {best_model['Prediction']}

Model : {best_model['Model']}

Confidence : {float(best_model['Confidence']):.2f}%
"""
        )

        # =========================
        # GRADCAM VISUALIZATION
        # =========================

        st.subheader("Grad-CAM Visualization")

        g1, g2, g3 = st.columns(3)

        with g1:
            st.image(
                data["original"],
                caption="MRI Image",
                width=180
            )

        with g2:
            st.image(
                data["heatmap"],
                caption="Heatmap",
                width=180
            )

        with g3:
            st.image(
                data["overlay"],
                caption="Grad-CAM Overlay",
                width=180
            )

    else:

        st.info(
            "Upload MRI image and click Analyze MRI."
        )