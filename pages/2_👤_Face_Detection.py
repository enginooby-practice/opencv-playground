import streamlit as st
import cv2
import numpy as np

st.title("👤 Face Detection")
st.write("Automatically detect faces in images")

# Load face detector
@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector = load_detector()

# Upload image
uploaded_file = st.file_uploader("Upload an image with faces", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detection settings
    st.subheader("Detection Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        scale = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01,
                         help="Lower = more accurate but slower")
    with col2:
        neighbors = st.slider("Min Neighbors", 1, 10, 5,
                             help="Higher = fewer false detections")
    
    # Detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
    
    # Draw boxes around faces
    img_result = img_rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_result, 'Face', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show results
    st.subheader("Results")
    
    if len(faces) > 0:
        st.success(f"✅ Found {len(faces)} face(s)!")
    else:
        st.warning("⚠️ No faces found. Try adjusting the settings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.write("**Detected Faces**")
        st.image(img_result, use_container_width=True)
    
    # Face details
    if len(faces) > 0:
        with st.expander("📋 Face Details"):
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"**Face {i+1}:** Position ({x}, {y}), Size {w}x{h}")

else:
    st.info("👆 Upload an image with faces!")
    st.write("""
    **How it works:**
    - Uses a pre-trained model to find faces
    - Draws green boxes around detected faces
    - Works best with frontal faces and good lighting
    """)