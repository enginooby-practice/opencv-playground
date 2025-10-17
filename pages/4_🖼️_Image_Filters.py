import streamlit as st
import cv2
import numpy as np

st.title("🖼️ Image Filters")
st.write("Apply cool effects to your images")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Select filters
    st.subheader("Select Filters")
    filters = st.multiselect(
        "Choose one or more:",
        ["Grayscale", "Blur", "Sharpen", "Sepia", "Cartoon"],
        default=["Grayscale"]
    )
    
    # Filter settings
    if "Blur" in filters:
        blur_amount = st.slider("Blur Amount", 1, 50, 15)
    if "Sharpen" in filters:
        sharpen_amount = st.slider("Sharpen Amount", 1, 10, 5)
    
    # Store results
    results = {"Original": img_rgb}
    
    # Apply filters
    if "Grayscale" in filters:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results["Grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    if "Blur" in filters:
        kernel = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        blurred = cv2.GaussianBlur(img_rgb, (kernel, kernel), 0)
        results["Blur"] = blurred
    
    if "Sharpen" in filters:
        kernel = np.array([[-1, -1, -1],
                          [-1, 9 + sharpen_amount, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img_rgb, -1, kernel)
        results["Sharpen"] = sharpened
    
    if "Sepia" in filters:
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img_rgb, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        results["Sepia"] = sepia
    
    if "Cartoon" in filters:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_rgb, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        results["Cartoon"] = cartoon
    
    # Display results
    st.subheader("Results")
    
    # Show in grid (2 per row)
    items = list(results.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(items):
                name, img_filtered = items[i + j]
                with cols[j]:
                    st.write(f"**{name}**")
                    st.image(img_filtered, use_container_width=True)

else:
    st.info("👆 Upload an image to apply filters!")
    st.write("""
    **Available filters:**
    - **Grayscale:** Black and white
    - **Blur:** Smooth and soft
    - **Sharpen:** More details
    - **Sepia:** Vintage brown tone
    - **Cartoon:** Cartoon-style effect
    
    **Tip:** Select multiple filters to compare!
    """)