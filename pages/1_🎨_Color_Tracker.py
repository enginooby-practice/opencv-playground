import streamlit as st
import cv2
import numpy as np

st.title("🎨 Color Tracker")
st.write("Track objects by their color")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read the image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Color range sliders
    st.subheader("Adjust Color Range")
    
    col1, col2 = st.columns(2)
    with col1:
        hue_min = st.slider("Hue Min", 0, 179, 30)
        sat_min = st.slider("Saturation Min", 0, 255, 50)
        val_min = st.slider("Value Min", 0, 255, 50)
    
    with col2:
        hue_max = st.slider("Hue Max", 0, 179, 90)
        sat_max = st.slider("Saturation Max", 0, 255, 255)
        val_max = st.slider("Value Max", 0, 255, 255)
    
    # Quick color presets
    st.write("**Quick presets:**")
    col_a, col_b, col_c, col_d = st.columns(4)
    if col_a.button("🟢 Green"):
        hue_min, hue_max = 35, 85
    if col_b.button("🔵 Blue"):
        hue_min, hue_max = 90, 130
    if col_c.button("🔴 Red"):
        hue_min, hue_max = 0, 10
    if col_d.button("🟡 Yellow"):
        hue_min, hue_max = 20, 40
    
    # Convert to HSV and create mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Show results
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original**")
        st.image(img_rgb, use_container_width=True)
    
    with col2:
        st.write("**Mask**")
        st.image(mask, use_container_width=True)
    
    with col3:
        st.write("**Tracked Color**")
        st.image(result_rgb, use_container_width=True)
    
    # Statistics
    pixels = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    percent = (pixels / total) * 100
    st.success(f"✅ Found {pixels:,} pixels ({percent:.1f}% of image)")

else:
    st.info("👆 Upload an image to start tracking colors!")
    st.write("""
    **How it works:**
    - Converts image to HSV color space
    - Creates a mask for colors in your range
    - Shows only the tracked color
    """)