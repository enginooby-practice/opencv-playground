import streamlit as st
import cv2
import numpy as np

st.title("🔍 Edge Detection")
st.write("Find edges and outlines in images")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection settings
    st.subheader("Edge Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        threshold1 = st.slider("Lower Threshold", 0, 500, 100)
    with col2:
        threshold2 = st.slider("Upper Threshold", 0, 500, 200)
    
    # Options
    blur = st.checkbox("Apply Blur (reduces noise)", value=True)
    
    # Process image
    processed = gray.copy()
    if blur:
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(processed, threshold1, threshold2)
    
    # Show results
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.write("**Detected Edges**")
        st.image(edges, use_container_width=True)
    
    # Statistics
    edge_pixels = cv2.countNonZero(edges)
    total = edges.shape[0] * edges.shape[1]
    percent = (edge_pixels / total) * 100
    st.info(f"📊 Edge pixels: {edge_pixels:,} ({percent:.1f}% of image)")

else:
    st.info("👆 Upload an image to detect edges!")
    st.write("""
    **How it works:**
    - Converts image to grayscale
    - Finds areas where brightness changes quickly
    - These changes are the edges!
    
    **Tips:**
    - Lower threshold = more edges detected
    - Upper threshold = only strong edges
    - Use blur to reduce noise
    """)