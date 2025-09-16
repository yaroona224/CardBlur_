# streamlit_app.py — CardBlur (Streamlit + YOLO, upload, camera, live)
import os, io, tempfile, urllib.request, pathlib
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Optional live cam
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ------------------
# Config & constants
# ------------------
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
CONF     = float(os.getenv("CONF", 0.28))
IOU      = float(os.getenv("IOU", 0.5))

DOC_KEYS  = ("id", "id_card", "idcard", "passport", "mrz", "document", "card")
FACE_KEYS = ("face", "person_face", "head")
TEXT_KEYS = ("text",)

# -------------
# UI: Sidebar
# -------------
st.set_page_config(page_title="CardBlur", page_icon="🪪")
st.title("🪪 CardBlur — ID/Passport Privacy Blurring")

with st.sidebar:
    st.header("Options")
    mode = st.radio(
        "What to blur?",
        ["Text only", "Face only", "Text + Face", "Whole card"],
        index=2
    )
    st.caption("Tip: If your custom model labels differ, I’ll try to auto-match them.")

# -----------------
# Weights utilities
# -----------------
def ensure_weights() -> str:
    """
    Returns a local path to best.pt.
    Looks for ./best.pt, ./models/best.pt, or downloads from st.secrets['WEIGHTS_URL'].
    """
    candidates = [
        pathlib.Path("best.pt"),
        pathlib.Path("models/best.pt"),
        pathlib.Path("weights/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())

    # Try secrets URL
    url = st.secrets.get("WEIGHTS_URL") if hasattr(st, "secrets") else None
    if url:
        cache_dir = pathlib.Path.home() / ".cache" / "cardblur"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / "best.pt"
        if not dst.exists():
            with st.status("Downloading model weights…"):
                urllib.request.urlretrieve(url, dst)
        return str(dst.resolve())

    raise FileNotFoundError(
        "best.pt not found. Put it at repo root or models/best.pt, or set WEIGHTS_URL in secrets."
    )

@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model

# -------------
# Helpers
# -------------
def make_odd(n:int)->int: return n if n % 2 else n + 1
def kernel_for_box(w:int, h:int, kscale:float=0.22, kmin:int=31)->int:
    k = max(int(max(w,h)*kscale), kmin)
    return make_odd(max(k,3))

def blur_region(img:np.ndarray, x1:int, y1:int, x2:int, y2:int):
    H, W = img.shape[:2]
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2));  y2 = max(0, min(H,   y2))
    if x2 <= x1 or y2 <= y1: return
    k = kernel_for_box(x2-x1, y2-y1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def run_yolo(model, bgr: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], str, float]]:
    # model expects RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(rgb, imgsz=IMG_SIZE, conf=CONF, iou=IOU, verbose=False)
    out = []
    names = model.model.names if hasattr(model, "model") else model.names
    for r in res:
        if r.boxes is None: 
            continue
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            cls = int(b.cls.item()) if b.cls is not None else -1
            score = float(b.conf.item()) if b.conf is not None else 0.0
            label = str(names.get(cls, str(cls))) if isinstance(names, dict) else str(cls)
            out.append(((x1,y1,x2,y2), label.lower(), score))
    return out

def select_targets(preds, names_map, chosen: str) -> List[Tuple[int,int,int,int]]:
    doc_labels  = {n for n in names_map if any(k in n for k in DOC_KEYS)}
    face_labels = {n for n in names_map if any(k in n for k in FACE_KEYS)}
    text_labels = {n for n in names_map if any(k in n for k in TEXT_KEYS)}

    boxes_doc  = [(b,label) for b,label,_ in preds if label in doc_labels]
    boxes_face = [(b,label) for b,label,_ in preds if label in face_labels]
    boxes_text = [(b,label) for b,label,_ in preds if label in text_labels]

    if chosen == "Whole card":
        return [b for b,_ in boxes_doc]
    if chosen == "Face only":
        return [b for b,_ in boxes_face]
    if chosen == "Text only":
        return [b for b,_ in boxes_text]
    # Text + Face
    return [b for b,_ in boxes_text] + [b for b,_ in boxes_face]

def process_image(model, bgr: np.ndarray, mode: str) -> np.ndarray:
    preds = run_yolo(model, bgr)
    names_set = {label for _,label,_ in preds}
    targets = select_targets(preds, names_set, mode)
    for (x1,y1,x2,y2) in targets:
        blur_region(bgr, x1, y1, x2, y2)
    return bgr

# -----------------
# Load model once
# -----------------
try:
    weights = ensure_weights()
    model = load_model(weights)
    names = model.model.names if hasattr(model, "model") else model.names
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# -----------------
# Image / snapshot
# -----------------
tab_upload, tab_camera, tab_live = st.tabs(["📁 Upload", "📷 Snapshot", "🎥 Live (WebRTC)"])

with tab_upload:
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
    if file:
        pil = Image.open(file).convert("RGB")
        bgr = pil_to_bgr(pil)
        out = process_image(model, bgr.copy(), mode)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Blurred", use_column_width=True)
        # Download button
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            st.download_button("Download blurred image", buf.tobytes(), file_name="cardblur_result.jpg", mime="image/jpeg")

with tab_camera:
    snap = st.camera_input("Take a snapshot")
    if snap:
        pil = Image.open(io.BytesIO(snap.getvalue())).convert("RGB")
        bgr = pil_to_bgr(pil)
        out = process_image(model, bgr.copy(), mode)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Blurred", use_column_width=True)

with tab_live:
    if not HAS_WEBRTC:
        st.info("Install streamlit-webrtc to enable true live video.")
    else:
        st.caption("Allow camera permissions when prompted.")
        rtc_cfg = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        class Transformer(VideoTransformerBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                out = process_image(model, img, mode)
                return type(frame)(out, format="bgr24")

        webrtc_streamer(
            key="cardblur-live",
            mode="SENDRECV",
            rtc_configuration=rtc_cfg,
            media_stream_constraints={"video": True, "audio": False},
            video_transformer_factory=Transformer,
        )

st.caption("Model labels detected: " + ", ".join([str(v) for v in (names.values() if isinstance(names, dict) else names)]))
