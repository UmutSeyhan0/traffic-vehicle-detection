import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="YOLO AI Panel", layout="wide", page_icon="🤖")


model_path = ""

train_folder_path = os.path.dirname(os.path.dirname(model_path))


@st.cache_resource
def load_model():
    return YOLO(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()


tab1, tab2 = st.tabs(["🚀 Test Application", "ℹ️ Training Data & Statistics"])


with tab1:
    st.sidebar.title("Test Settings")
    mode = st.sidebar.radio("Test Mode", ["Image", "Video"])

    if mode == "Image":
        st.write("### 📸 Image Testing")
        uploaded_file = st.file_uploader(
            "Upload an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            if st.button("Start Detection"):
                res = model(image)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption="Result", use_column_width=True)

    elif mode == "Video":
        st.write("### 🎥 Video Testing")
        uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            if st.button("Process Video"):
                st_frame = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame)
                    res_plotted = results[0].plot()
                    st_frame.image(res_plotted, channels="BGR")
                cap.release()


with tab2:
    st.header("📊 Model Training Data")
    st.write(
        f"This model uses training results from the **{os.path.basename(train_folder_path)}** folder."
    )

    labels_path = os.path.join(train_folder_path, "labels.jpg")

    if not os.path.exists(labels_path):
        labels_path = os.path.join(train_folder_path, "labels.png")

    col_stats1, col_stats2 = st.columns(2)

    with col_stats1:
        st.subheader("Dataset Distribution (Class Counts)")
        if os.path.exists(labels_path):
            st.image(
                labels_path,
                caption="Training Data Class Distribution",
                use_column_width=True,
            )
            st.info(
                "👆 This chart shows how many samples were used for each vehicle class during training."
            )
        else:
            st.warning(
                f"⚠️ 'labels.jpg' file not found. Please check the folder: {train_folder_path}"
            )

    cm_path = os.path.join(train_folder_path, "confusion_matrix.png")

    with col_stats2:
        st.subheader("Confusion Matrix (Model Errors)")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
            st.info(
                "👆 This chart shows which vehicle classes the model tends to confuse."
            )
        else:
            st.warning("⚠️ 'confusion_matrix.png' file not found.")

    st.divider()

    results_path = os.path.join(train_folder_path, "results.png")
    st.subheader("📈 Training Performance Graph")
    if os.path.exists(results_path):
        st.image(results_path, use_column_width=True)
    else:
        st.write("Training results graph not found.")
