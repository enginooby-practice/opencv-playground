import streamlit as st
import cv2
import numpy as np

st.title("👤 Face Detection")
st.write("Automatically detect faces in images")


# -------------------------------
# 🤖 Load the Haar Cascade Face Detector
# -------------------------------
# Haar cascades are pre-trained models stored as XML files in OpenCV.
# They can detect faces, eyes, smiles, etc.
# We cache it so Streamlit doesn’t reload it every time the script runs.
@st.cache_resource
def load_detector():
    """Load OpenCV's pre-trained frontal face detector."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

detector = load_detector()

# -------------------------------
# 📸 Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an image with faces", type=['jpg', 'jpeg', 'png'])

# -------------------------------
# 📷 When the user uploads an image
# -------------------------------
if uploaded_file:
    # Convert the uploaded file (binary) into a NumPy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    
    # Decode bytes to an OpenCV image (same as reading from disk)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert color spaces:
    # BGR → RGB (for Streamlit display)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # BGR → GRAY (for face detection, which works on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # -------------------------------
    # ⚙️ Detection Settings (User Adjustable)
    # -------------------------------
    st.subheader("Detection Settings")
    
    col1, col2 = st.columns(2)

    with col1:
        # Scale factor controls how much the image is reduced at each image scale.
        # Lower = more detailed search (slower, more accurate)
        scale = st.slider(
            "Scale Factor", 1.01, 1.5, 1.1, 0.01,
            help="Lower values = more accurate but slower detection."
        )
    with col2:
        # Min neighbors defines how many detections a rectangle needs to be considered a face.
        # Higher = fewer false positives, but might miss small faces.
        neighbors = st.slider(
            "Min Neighbors", 1, 10, 5,
            help="Higher values = fewer false detections, but may miss faces."
        )
    
    # -------------------------------
    # 🧩 Detect Faces in the Image
    # -------------------------------
    faces = detector.detectMultiScale(
        gray, 
        scaleFactor=scale, 
        minNeighbors=neighbors
    )
    
    # -------------------------------
    # 🖍️ Draw Boxes Around Detected Faces
    # -------------------------------
    img_result = img_rgb.copy()
    for (x, y, w, h) in faces:
        # Draw a green rectangle around each detected face
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # Add label text "Face" above the box
        cv2.putText(
            img_result, 'Face', (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    
    # -------------------------------
    # 🖼️ Display Results
    # -------------------------------
    st.subheader("Results")
    
    # Display how many faces were found
    if len(faces) > 0:
        st.success(f"✅ Found {len(faces)} face(s)!")
    else:
        st.warning("⚠️ No faces found. Try adjusting the detection settings.")

    # Side-by-side view: Original vs Detected
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Image**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.write("**Detected Faces**")
        st.image(img_result, use_container_width=True)
    
    # -------------------------------
    # 📋 Show Face Details (Optional)
    # -------------------------------
    if len(faces) > 0:
        with st.expander("📋 Face Details"):
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"**Face {i+1}:** Position ({x}, {y}), Size {w}×{h}")

# -------------------------------
# ℹ️ If No Image Uploaded Yet
# -------------------------------
else:
    st.info("👆 Upload an image with faces to start detection!")
    st.write("""
    ### How it works:
    1. Converts your image to **grayscale**
    2. Uses a **Haar Cascade** pre-trained model to detect faces
    3. Draws green rectangles around detected faces

    ⚙️ Tips:
    - Works best with **frontal faces** and **good lighting**
    - If faces are missed, lower the *Scale Factor* or *Min Neighbors*
    """)