import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd
from PIL import Image
data=pd.read_csv('owner_data.csv')
st.title("Number Plate Recognition System")

# Load EasyOCR for text extraction
reader = easyocr.Reader(['en'])

# Load database of vehicle owners


owner_data = data

# Camera Input
st.sidebar.title("Camera Feed")
run = st.sidebar.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image!")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plate (Simple method)
    plates_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plates_cascade.detectMultiScale(gray, 1.1, 10)

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        text = reader.readtext(plate_img, detail=0)
        plate_number = text[0] if text else "Not detected"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, plate_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Lookup owner details
        owner_info = owner_data[owner_data["Number Plate"] == plate_number]
        if not owner_info.empty:
            st.sidebar.write(f"*Owner:* {owner_info['Owner'].values[0]}")
            st.sidebar.write(f"*Car Model:* {owner_info['Car Model'].values[0]}")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
