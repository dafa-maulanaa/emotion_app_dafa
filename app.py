import os
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import joblib
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import matplotlib.pyplot as plt
import altair as alt

# ----------------- CONFIG -----------------
MODEL_PATH = "emotion_model.keras"       # Pastikan file model berada di sini
LABEL_ENCODER_PATH = "label_encoder.pkl"  # Pastikan file encoder berada di sini
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174
# ------------------------------------------

# ----------------- Load Model & Encoder -----------------
model = None
le = None
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)  # Memuat model dari direktori saat ini
        le = joblib.load(LABEL_ENCODER_PATH)  # Memuat encoder dari direktori saat ini
        st.success("Model dan Label Encoder dimuat.")
    except Exception as e:
        st.error("Gagal memuat model/encoder.")
        st.exception(e)
else:
    st.warning("Model atau encoder tidak ditemukan. Jalankan train_model.py dulu.")
# --------------------------------------------------------

# -------------- Audio Processor Using WebRTC -------------
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        audio_data = np.array(frame).flatten()
        # Ekstraksi fitur MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            pad = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
        else:
            mfcc = mfcc[:, :MAX_LEN]
        X = mfcc[np.newaxis, ...]
        X = X[..., np.newaxis]
        X = np.transpose(X, (0, 2, 1, 3))  # (batch, time, height, width)
        # Prediksi Emosi
        preds = model.predict(X)[0]
        idx = int(np.argmax(preds))
        label = le.inverse_transform([idx])[0]
        return label, preds
# --------------------------------------------------------

# -------------- Streamlit WebRTC Streamer --------------
st.title("Deteksi Emosi dari Suara")
st.write("Rekam suara dengan klik tombol di bawah ini")

# Cek apakah WebRTC tersedia
try:
    webrtc_ctx = webrtc_streamer(
        key="audio-emotion",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.audio_receiver:
        label, probs = webrtc_ctx.audio_receiver.recv()
        st.write(f"Prediksi Emosi: {label}")
        st.write(f"Probabilitas: {probs}")
except Exception as e:
    st.error("WebRTC tidak dapat digunakan. Pastikan aplikasi memiliki akses mikrofon atau coba unggah file audio.")
    st.exception(e)

# ===================== History Prediksi =================
st.subheader("History Prediksi")
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp", "file", "label", "probs"])

# Alternatif: Gunakan unggah file jika WebRTC gagal
uploaded = st.file_uploader("Unggah file audio (hanya .wav)", type=["wav"])
if uploaded is not None:
    audio, sr = sf.read(uploaded)
    # Ekstraksi fitur MFCC dan prediksi
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]
    X = mfcc[np.newaxis, ...]
    X = X[..., np.newaxis]
    X = np.transpose(X, (0, 2, 1, 3))  # (batch, time, height, width)
    preds = model.predict(X)[0]
    idx = int(np.argmax(preds))
    label = le.inverse_transform([idx])[0]
    st.write(f"Prediksi Emosi: {label}")
    st.write(f"Probabilitas: {preds}")

    # Simpan riwayat
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_data = {
        "timestamp": timestamp,
        "file": uploaded.name,
        "label": label,
        "probs": str(preds)
    }
    history_df = pd.DataFrame([history_data])
    st.session_state.history = pd.concat([st.session_state.history, history_df], ignore_index=True)
    st.write(st.session_state.history.tail(10))  # Display the last 10 prediction results
# =========================================================
