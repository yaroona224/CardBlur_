import streamlit as st
from PIL import Image

st.set_page_config(page_title="CardBlur", page_icon="🪪")
st.title("🪪 CardBlur")

img_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

st.caption("Made by Yara & Shatha 💜")
