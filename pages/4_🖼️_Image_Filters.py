import streamlit as st
import cv2
import numpy as np

st.title("🖼️ Image Filters")
st.write("Apply cool effects to your images")

# -------------------------------
# 📸 Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# -------------------------------
# 📷 When user uploads an image
# -------------------------------
if uploaded_file:
    # Convert uploaded file (binary data) into a NumPy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    
    # Decode the byte array as an image (same as reading with cv2.imread)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert from BGR (OpenCV default) → RGB (for Streamlit display)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # -------------------------------
    # 🎛️ Choose Filters
    # -------------------------------
    st.subheader("Select Filters to Apply")
    filters = st.multiselect(
        "Choose one or more filters to apply:",
        ["Grayscale", "Blur", "Sharpen", "Sepia", "Cartoon"],
        default=["Grayscale"]
    )
    
    # -------------------------------
    # ⚙️ Filter Parameters (if needed)
    # -------------------------------
    if "Blur" in filters:
        # Blur strength — higher = smoother image
        blur_amount = st.slider("Blur Amount", 1, 50, 15)
    if "Sharpen" in filters:
        # Sharpen intensity — higher = stronger detail enhancement
        sharpen_amount = st.slider("Sharpen Amount", 1, 10, 5)
    
    # A dictionary to store filtered results
    results = {"Original": img_rgb}

    # -------------------------------
    # 🧩 APPLY SELECTED FILTERS
    # -------------------------------

    # 1️⃣ GRAYSCALE — Convert image to black & white
    if "Grayscale" in filters:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert back to RGB for consistent display in Streamlit
        results["Grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 2️⃣ BLUR — Smooth the image using a Gaussian blur
    if "Blur" in filters:
        # Kernel size must be odd; add +1 if even
        kernel = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        blurred = cv2.GaussianBlur(img_rgb, (kernel, kernel), 0)
        results["Blur"] = blurred

    # 3️⃣ SHARPEN — Enhance image edges using a sharpening kernel
    if "Sharpen" in filters:
        # Create a custom sharpening kernel (filter matrix)
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9 + sharpen_amount, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(img_rgb, -1, kernel)
        results["Sharpen"] = sharpened

    # 4️⃣ SEPIA — Add warm brown tones (vintage look)
    if "Sepia" in filters:
        sepia_filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(img_rgb, sepia_filter)
        # Ensure values stay in [0,255] range and are uint8 type
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        results["Sepia"] = sepia

    # 5️⃣ CARTOON — Make the image look like a cartoon
    if "Cartoon" in filters:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduce noise before edge detection
        gray = cv2.medianBlur(gray, 5)
        # Detect edges (binary mask)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        # Apply bilateral filter to smooth colors while keeping edges
        color = cv2.bilateralFilter(img_rgb, 9, 300, 300)
        # Combine smoothed colors with edge mask
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        results["Cartoon"] = cartoon

    # -------------------------------
    # 🖼️ Display Filter Results
    # -------------------------------
    st.subheader("Results")

    # Display results in a grid (2 images per row)
    items = list(results.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(items):
                name, img_filtered = items[i + j]
                with cols[j]:
                    st.write(f"**{name}**")
                    st.image(img_filtered, use_container_width=True)

# -------------------------------
# ℹ️ If No Image Uploaded Yet
# -------------------------------
else:
    st.info("👆 Upload an image to start applying filters!")
    st.write("""
    ### 🧰 Available Filters:
    - **Grayscale:** Convert image to black and white  
    - **Blur:** Smooth and soften edges  
    - **Sharpen:** Enhance image details  
    - **Sepia:** Apply a warm, vintage look  
    - **Cartoon:** Turn your photo into a cartoon-like drawing  

    💡 **Tip:** You can select multiple filters at once to compare their effects side-by-side.
    """)