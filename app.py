import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import os

from ultralytics import YOLO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from PIL import Image

os.makedirs("temp", exist_ok=True)

SEQUENCE_LENGTH = 10
IMG_SIZE = 224

st.set_page_config(
page_title="Multimodal Accident Risk Assessment",
layout="wide"
)

st.title(
"Multimodal Accident Risk Assessment System"
)

@st.cache_resource
def load_models():

```
rf_model = joblib.load(
    "models/csv_risk_model.pkl"
)

yolo_model = YOLO(
    "models/best.pt"
)

resnet = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224,224,3)
)

resnet.trainable = False

video_model = load_model(
    "models/video_accident_model.h5"
)

return (
    rf_model,
    yolo_model,
    resnet,
    video_model
)
```

rf_model, yolo_model, resnet, video_model = load_models()

def extract_frames(video_path):

```
cap = cv2.VideoCapture(video_path)

frames = []

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if total_frames <= 0:

    cap.release()

    return None

frame_indices = np.linspace(
    0,
    total_frames - 1,
    SEQUENCE_LENGTH,
    dtype=int
)

idx = 0

selected = set(frame_indices)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if idx in selected:

        frame = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    idx += 1

cap.release()

if len(frames) != SEQUENCE_LENGTH:
    return None

return np.array(frames)
```

def extract_features(frames):

```
frames = preprocess_input(frames)

features = resnet.predict(
    frames,
    verbose=0
)

return features
```

st.header("Upload Inputs")

csv_file = st.file_uploader(
"Upload Environmental CSV",
type=["csv"]
)

image_file = st.file_uploader(
"Upload Traffic Sign Image",
type=["jpg","jpeg","png"]
)

video_file = st.file_uploader(
"Upload Accident Video",
type=["mp4","avi","mov"]
)

if st.button("Analyze"):

```
if csv_file is None:

    st.error(
        "Upload CSV file"
    )

    st.stop()

if image_file is None:

    st.error(
        "Upload Image file"
    )

    st.stop()

if video_file is None:

    st.error(
        "Upload Video file"
    )

    st.stop()

try:

    df = pd.read_csv(
        csv_file
    )

    csv_score = float(
        rf_model.predict(df)[0]
    )

    image_path = "temp/test.jpg"

    with open(image_path,"wb") as f:

        f.write(
            image_file.read()
        )

    results = yolo_model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) > 0:

        sign_score = float(
            boxes.conf.max().cpu().numpy()
        )

        class_id = int(
            boxes.cls[0].cpu().numpy()
        )

        sign_name = yolo_model.names[class_id]

        annotated = results[0].plot()

    else:

        sign_score = 0.0

        sign_name = "No Sign Detected"

        annotated = Image.open(
            image_path
        )

    video_path = "temp/test_video.mp4"

    with open(video_path,"wb") as f:

        f.write(
            video_file.read()
        )

    frames = extract_frames(
        video_path
    )

    if frames is None:

        st.error(
            "Frame Extraction Failed"
        )

        st.stop()

    features = extract_features(
        frames
    )

    features = np.expand_dims(
        features,
        axis=0
    )

    prediction = video_model.predict(
        features,
        verbose=0
    )

    video_score = float(
        prediction[0][0]
    )

    if video_score > 0.5:

        video_label = "ACCIDENT"

    else:

        video_label = "NORMAL"

    final_score = (
        0.3 * csv_score +
        0.3 * sign_score +
        0.4 * video_score
    )

    if final_score > 0.7:

        risk = "HIGH"

    elif final_score > 0.4:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    st.subheader(
        "Results"
    )

    col1,col2,col3,col4 = st.columns(4)

    with col1:

        st.metric(
            "CSV Score",
            round(csv_score,4)
        )

    with col2:

        st.metric(
            "Sign Score",
            round(sign_score,4)
        )

    with col3:

        st.metric(
            "Video Score",
            round(video_score,4)
        )

    with col4:

        st.metric(
            "Risk Level",
            risk
        )

    st.write(
        f"Traffic Sign : {sign_name}"
    )

    st.write(
        f"Video Prediction : {video_label}"
    )

    st.write(
        f"Final Risk Score : {final_score:.4f}"
    )

    st.subheader(
        "Traffic Sign Detection"
    )

    st.image(
        annotated
    )

    st.subheader(
        "Video Frame Used"
    )

    st.image(
        frames[4],
        caption=f"{video_label} | Probability: {video_score:.4f}"
    )

except Exception as e:

    st.error(
        f"Prediction Error : {e}"
    )
```
