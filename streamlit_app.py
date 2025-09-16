import cv2, numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="CardBlur", page_icon="🪪")
st.title("🪪 CardBlur — MVP")

# --- load model once ---
@st.cache_resource(show_spinner=True)
def load_model():
    from ultralytics import YOLO
    return YOLO("best.pt")  # assumes best.pt is in repo root

model = load_model()
st.success("Model loaded ✅")

# --- helpers ---
def make_odd(n): return n if n % 2 else n + 1
def kernel_for_box(w,h,kscale=0.22,kmin=31):
    k = max(int(max(w,h)*kscale), kmin)
    k = make_odd(max(k,3))
    return (k,k)

def blur_box(bgr, x1,y1,x2,y2):
    H,W = bgr.shape[:2]
    x1 = max(0,min(W-1,x1)); y1 = max(0,min(H-1,y1))
    x2 = max(0,min(W,  x2)); y2 = max(0,min(H,  y2))
    if x2<=x1 or y2<=y1: return
    k = kernel_for_box(x2-x1, y2-y1)
    bgr[y1:y2, x1:x2] = cv2.GaussianBlur(bgr[y1:y2, x1:x2], k, 0)

def run_and_blur(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = model.predict(bgr, imgsz=640, conf=0.28, iou=0.5, verbose=False)

    # handle both normal YOLO and OBB models safely
    for r in res:
        if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                blur_box(bgr, x1,y1,x2,y2)
        elif getattr(r, "obb", None) is not None and r.obb is not None:
            # try to use obb.xyxy if present; otherwise fall back to polygon -> aabb
            if hasattr(r.obb, "xyxy"):
                for p in r.obb.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = map(int, p.tolist())
                    blur_box(bgr, x1,y1,x2,y2)
            elif hasattr(r.obb, "xyxyn"):
                H,W = bgr.shape[:2]
                for p in r.obb.xyxyn.cpu().numpy():
                    x1,y1,x2,y2 = p.tolist()
                    blur_box(bgr, int(x1*W), int(y1*H), int(x2*W), int(y2*H))
            # if API differs, we simply skip obb gracefully

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)

# --- UI ---
img_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if img_file:
    pil = Image.open(img_file).convert("RGB")
    result = run_and_blur(pil)
    st.image(result, caption="Blurred", use_column_width=True)
    # download
    import io
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=95)
    st.download_button("Download blurred image", buf.getvalue(), file_name="cardblur.jpg", mime="image/jpeg")

st.caption("Made by Yara & Shatha 💜")
