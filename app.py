import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

model = pickle.load(open("model.pkl", "rb"))

labels = ['A','B','C','D','E','F','G','H','I','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y']

st.set_page_config(page_title="Sign Language Recognition", layout="wide")

st.title("🤟 Sign Language Recognition")
st.markdown("---")

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Camera Feed")
    img_file = st.camera_input("Show your hand sign!")

with col2:
    st.subheader("🔤 Detection")
    
    if img_file is not None:
        # Image process பண்ணு
        img = Image.open(img_file)
        img_array = np.array(img)
        
        # BGR convert
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # ROI - center crop
        h, w = frame.shape[:2]
        x1, y1 = w//4, h//4
        x2, y2 = 3*w//4, 3*h//4
        
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        flattened = resized.flatten().reshape(1, -1)
        
        prediction = model.predict(flattened)[0]
        letter = labels[prediction] if prediction < len(labels) else str(prediction)
        
        st.markdown(f"## Detected Letter: `{letter}`")
        st.markdown(f"## Sentence: `{st.session_state.sentence}`")
        
        # Add to sentence
        if st.button("✅ Add Letter"):
            st.session_state.sentence += letter
            st.rerun()
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.sentence = ""
            st.rerun()
    with col4:
        if st.button("➕ Space"):
            st.session_state.sentence += " "
            st.rerun()
    
    st.markdown(f"### 📝 Sentence: `{st.session_state.sentence}`")