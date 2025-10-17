import streamlit as st
import cv2
import numpy as np

st.title("🔍 Edge Detection")
st.write("Find edges and outlines in images")

# -------------------------------
# 📸 Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# -------------------------------
# 📷 When the user uploads an image
# -------------------------------
if uploaded_file:
    # Convert the uploaded file (binary data) into a NumPy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    
    # Decode bytes to an OpenCV image (like reading a file with cv2.imread)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert color spaces:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # for display in Streamlit
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # for edge detection
    
    # -------------------------------
    # ⚙️ Edge Detection Settings
    # -------------------------------
    st.subheader("Edge Detection Settings")

    # Two threshold values control edge sensitivity
    # Lower threshold → detects more edges (including weaker ones)
    # Upper threshold → detects only stronger edges
    col1, col2 = st.columns(2)
    with col1:
        threshold1 = st.slider("Lower Threshold", 0, 500, 100)
    with col2:
        threshold2 = st.slider("Upper Threshold", 0, 500, 200)
    
    # Checkbox to optionally apply blur before edge detection
    blur = st.checkbox("Apply Gaussian Blur (reduce noise)", value=True)
    
    # -------------------------------
    # 🧩 Image Preprocessing
    # -------------------------------
    processed = gray.copy()

    if blur:
        # Gaussian blur smooths small variations to prevent noise edges
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
    
    # -------------------------------
    # 🧠 Edge Detection (Canny Algorithm)
    # -------------------------------
    # The Canny algorithm finds areas in the image where intensity changes sharply.
    edges = cv2.Canny(processed, threshold1, threshold2)
    
    # -------------------------------
    # 🖼️ Display Results
    # -------------------------------
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Image**")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.write("**Detected Edges (Canny Output)**")
        st.image(edges, use_container_width=True)
    
    # -------------------------------
    # 📊 Edge Statistics
    # -------------------------------
    edge_pixels = cv2.countNonZero(edges)  # Count non-black pixels (edges)
    total = edges.shape[0] * edges.shape[1]
    percent = (edge_pixels / total) * 100

    st.info(f"📊 Edge pixels detected: {edge_pixels:,} ({percent:.1f}% of the image)")

# -------------------------------
# ℹ️ If No Image Uploaded Yet
# -------------------------------
else:
    st.info("👆 Upload an image to detect edges!")
    st.write("""
    ### How it works:
    1. Converts the image to **grayscale**
    2. Optionally applies a **Gaussian blur** to reduce noise
    3. Uses the **Canny Edge Detector** to find intensity changes
    4. Displays a binary image showing edges (white = edge)

    ### Tips:
    - Lower threshold → more edges (including weak ones)
    - Higher threshold → fewer, stronger edges
    - Enable *Blur* to remove small noise edges
    """)