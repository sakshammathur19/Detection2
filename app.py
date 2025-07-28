import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

st.title(" Object Detection App (YOLOv8)")
st.markdown("Detect objects in images or using your webcam")


option = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Webcam"))


if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting objects..."):
            results = model(img_array)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_array, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.image(img_array, caption="Detected Objects", use_column_width=True)


elif option == "Use Webcam":
    st.warning("Click below to start webcam (only works in local environment)")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Webcam not detected.")
        else:
            stframe = st.empty()

            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                
                results = model(frame)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
