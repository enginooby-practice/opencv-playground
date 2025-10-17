import streamlit as st

st.set_page_config(page_title="OpenCV Projects", page_icon="🎨", layout="wide")

st.title("🎨 OpenCV Mini Projects")
st.markdown("---")

st.markdown("""
### Welcome! 👋
Select a project from the **sidebar** to get started.

#### Available Projects:
1. **🎨 Color Tracker** - Track colors in images
2. **👤 Face Detection** - Find faces automatically
3. **🔍 Edge Detection** - Detect edges in images
4. **🖼️ Image Filters** - Apply cool effects

#### How to use:
1. Click a project in the sidebar
2. Upload an image
3. Play with the sliders
4. See the results!
""")

st.info("💡 Tip: Try different images to see how each project works!")