import streamlit as st
import cv2
import numpy as np

st.title("🎨 Color Tracker")
st.write("Track objects by their color")

# Upload an image file from the user's computer
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# -------------------------------
# 📷 If the user uploads an image
# -------------------------------
if uploaded_file:
    # Convert the uploaded file (binary) into a NumPy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    
    # Decode the array as an image (like reading it with cv2.imread)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert from BGR (OpenCV default) to RGB (for correct display in Streamlit)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # -------------------------------
    # 🎛️ Color Range Adjustment
    # -------------------------------
    st.subheader("Adjust Color Range (HSV)")

    # HSV means: Hue (color type), Saturation (color intensity), Value (brightness)
    col1, col2 = st.columns(2)

    # Minimum slider values (lower boundary of the color range)
    with col1:
        hue_min = st.slider("Hue Min", 0, 179, 30)
        sat_min = st.slider("Saturation Min", 0, 255, 50)
        val_min = st.slider("Value Min", 0, 255, 50)

    # Maximum slider values (upper boundary of the color range)
    with col2:
        hue_max = st.slider("Hue Max", 0, 179, 90)
        sat_max = st.slider("Saturation Max", 0, 255, 255)
        val_max = st.slider("Value Max", 0, 255, 255)
    
    # -------------------------------
    # 🎨 Quick Preset Buttons
    # -------------------------------
    st.write("**Quick presets:** Choose common color ranges")
    col_a, col_b, col_c, col_d = st.columns(4)

    # These buttons quickly set hue ranges for basic colors
    if col_a.button("🟢 Green"):
        hue_min, hue_max = 35, 85
    if col_b.button("🔵 Blue"):
        hue_min, hue_max = 90, 130
    if col_c.button("🔴 Red"):
        hue_min, hue_max = 0, 10
    if col_d.button("🟡 Yellow"):
        hue_min, hue_max = 20, 40
    
    # -------------------------------
    # 🧩 Create the Color Mask
    # -------------------------------
    # Convert image to HSV color space for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of your target color
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    
    # Create a binary mask:
    # White (255) where the color is in range, black (0) where it's not
    mask = cv2.inRange(hsv, lower, upper)
    
    # -------------------------------
    # 🔍 Apply Mask to Extract Color
    # -------------------------------
    # Keep only the pixels that match the mask
    result = cv2.bitwise_and(img, img, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # 📊 Display Results
    # -------------------------------
    st.subheader("Results Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Original Image**")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.write("**Mask (White = Selected Color)**")
        st.image(mask, use_container_width=True)

    with col3:
        st.write("**Tracked Color Output**")
        st.image(result_rgb, use_container_width=True)
    
    # -------------------------------
    # 📈 Simple Color Statistics
    # -------------------------------
    pixels = cv2.countNonZero(mask)  # number of non-black pixels (color found)
    total = mask.shape[0] * mask.shape[1]
    percent = (pixels / total) * 100

    # Show how much of the image is that color
    st.success(f"✅ Found {pixels:,} pixels ({percent:.1f}% of the image)")

# -------------------------------
# ℹ️ If no image is uploaded yet
# -------------------------------
else:
    st.info("👆 Upload an image above to start tracking colors!")
    st.write("""
    ### How it works:
    1. Converts your image to **HSV color space**
    2. Creates a **mask** of pixels inside your selected color range
    3. Shows only the pixels matching that color

    👉 Try adjusting the sliders to isolate a color or click on presets.
    """)