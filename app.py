import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

st.set_page_config(page_title="Deteksi Helm Pengendara Motor", page_icon="🏍️", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def hitung_kelas(result, names):
    """Hitung jumlah objek per kelas pada satu hasil prediksi."""
    counts = {n: 0 for n in names.values()}
    for c in result.boxes.cls.tolist():
        counts[names[int(c)]] += 1
    return counts


if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan di {MODEL_PATH}. Letakkan best.pt di folder models/.")
    st.stop()

model = load_model()

st.title("Deteksi Helm Pengendara Motor")
st.caption("Model YOLO11 yang dilatih pada dataset HELMET. Unggah gambar atau video untuk dideteksi.")

with st.sidebar:
    st.header("Pengaturan")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    imgsz = st.selectbox("Ukuran input (imgsz)", [640, 800, 960], index=2,
                         help="960 sama dengan ukuran saat training. Ukuran lebih kecil lebih cepat tetapi objek kecil bisa terlewat.")

tab_img, tab_vid = st.tabs(["Gambar", "Video"])

# ---------- Gambar ----------
with tab_img:
    file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], key="img")
    if file:
        img = Image.open(file).convert("RGB")
        bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
        with st.spinner("Mendeteksi..."):
            r = model.predict(bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]

        col1, col2 = st.columns(2)
        col1.image(img, caption="Gambar asli", use_container_width=True)
        col2.image(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), caption="Hasil deteksi",
                   use_container_width=True)

        counts = hitung_kelas(r, model.names)
        cols = st.columns(len(counts))
        for c, (nama, jml) in zip(cols, counts.items()):
            c.metric(nama, jml)

# ---------- Video ----------
with tab_vid:
    vid = st.file_uploader("Unggah video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    step = st.slider("Proses 1 frame tiap N frame", 1, 10, 3,
                     help="Nilai lebih besar membuat proses lebih cepat, tetapi tampilan kurang mulus.")

    if vid and st.button("Mulai deteksi"):
        suffix = Path(vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        tampil = st.empty()
        bar = st.progress(0.0)

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                r = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
                tampil.image(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            i += 1
            bar.progress(min(i / total, 1.0))

        cap.release()
        os.remove(tmp_path)
        st.success(f"Selesai. {i} frame dibaca.")
