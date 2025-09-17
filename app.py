import os
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
import joblib
from tensorflow.keras.models import load_model

# ----------------- CONFIG -----------------
MODEL_PATH = "models/emotion_model.keras"       # sesuai train_model.py
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
RECORDINGS_DIR = "recordings"
HISTORY_FILE = "prediction_history.csv"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174
MAX_RECORD_SECONDS = 10
# ------------------------------------------

os.makedirs(RECORDINGS_DIR, exist_ok=True)

st.set_page_config(page_title="Deteksi Emosi Suara", layout="centered", page_icon="")
st.title("Deteksi Emosi dari Suara")

# ---------- Load model & encoder ----------
model = None
le = None
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        le = joblib.load(LABEL_ENCODER_PATH)
        st.success("Model dan Label Encoder dimuat.")
    except Exception as e:
        st.error("Gagal memuat model/encoder.")
        st.exception(e)
else:
    st.warning("Model atau encoder tidak ditemukan. Jalankan train_model.py dulu.")

# ---------- Session state ----------
if "audio" not in st.session_state:
    st.session_state.audio = None
if "sr" not in st.session_state:
    st.session_state.sr = SAMPLE_RATE
if "pred" not in st.session_state:
    st.session_state.pred = None
if "pred_probs" not in st.session_state:
    st.session_state.pred_probs = None
if "audio_name" not in st.session_state:
    st.session_state.audio_name = None
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            st.session_state.history = pd.read_csv(HISTORY_FILE)
        except Exception:
            st.session_state.history = pd.DataFrame(columns=["timestamp", "file", "label", "probs"])
    else:
        st.session_state.history = pd.DataFrame(columns=["timestamp", "file", "label", "probs"])

# ---------- Helper functions ----------
def record_blocking(duration_sec: float = MAX_RECORD_SECONDS):
    try:
        duration_sec = float(duration_sec)
        if duration_sec <= 0 or duration_sec > 60:
            duration_sec = MAX_RECORD_SECONDS
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
        st.info(f"🎙 Merekam selama {duration_sec:.1f} detik...")
        audio = sd.rec(int(duration_sec * SAMPLE_RATE), dtype="float32")
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        st.error("Error saat merekam audio.")
        st.exception(e)
        return None

def extract_features_safe(audio: np.ndarray, sr: int):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    max_samples = int(MAX_RECORD_SECONDS * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]
    X = mfcc[np.newaxis, ...]
    X = X[..., np.newaxis]
    X = np.transpose(X, (0, 2, 1, 3))  # (batch, time, height, width)
    return X.astype(np.float32)

def predict_audio(audio: np.ndarray, sr: int):
    if model is None or le is None:
        st.error("Model belum siap.")
        return None, None
    X = extract_features_safe(audio, sr)
    preds = model.predict(X)[0]
    idx = int(np.argmax(preds))
    label = le.inverse_transform([idx])[0]
    return label, preds

# ---------- UI ----------
st.markdown("## Rekam atau Unggah Audio")
col_r1, col_r2 = st.columns([2,2])

with col_r1:
    dur = st.slider("Durasi rekaman (detik)", 1, MAX_RECORD_SECONDS, MAX_RECORD_SECONDS)
    if st.button("⏺ Rekam"):
        audio = record_blocking(dur)
        if audio is not None:
            st.session_state.audio = audio
            st.session_state.sr = SAMPLE_RATE
            st.session_state.audio_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            st.success("Rekaman selesai.")

with col_r2:
    uploaded = st.file_uploader("Unggah file .wav", type=["wav"])
    if uploaded is not None:
        try:
            data, sr = sf.read(io.BytesIO(uploaded.read()), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            st.session_state.audio = data
            st.session_state.sr = sr
            st.session_state.audio_name = uploaded.name
            st.success("File terunggah.")
        except Exception as e:
            st.error("Gagal membaca file audio.")
            st.exception(e)

if st.session_state.audio is not None:
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, st.session_state.audio, st.session_state.sr, format="WAV")
    st.audio(wav_bytes.getvalue(), format="audio/wav")

    if st.button("Prediksi Emosi"):
        label, probs = predict_audio(st.session_state.audio, st.session_state.sr)
        if label:
            st.session_state.pred = label
            st.session_state.pred_probs = probs
            st.success(f"Emosi Terprediksi: **{label}**")

            # Save prediction history
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            history_data = {
                "timestamp": timestamp,
                "file": st.session_state.audio_name,
                "label": label,
                "probs": str(probs)
            }
            history_df = pd.DataFrame([history_data])

            if os.path.exists(HISTORY_FILE):
                history_df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
            else:
                history_df.to_csv(HISTORY_FILE, index=False)

    if st.session_state.pred is not None:
        st.markdown("## Hasil Prediksi")
        st.write(f"**Label:** {st.session_state.pred}")
        dfp = pd.DataFrame({"emotion": le.classes_, "prob": st.session_state.pred_probs})
        chart = alt.Chart(dfp).mark_bar().encode(
            x="emotion:N", y="prob:Q", color="emotion:N"
        ).properties(width=600, height=300)
        st.altair_chart(chart)

        # Tambahkan visualisasi waveform & MFCC
        st.markdown("### Waveform")
        fig, ax = plt.subplots(figsize=(8,2))
        librosa.display.waveshow(st.session_state.audio, sr=st.session_state.sr, ax=ax)
        st.pyplot(fig)

        st.markdown("### MFCC")
        mfcc_vis = librosa.feature.mfcc(y=st.session_state.audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        fig2, ax2 = plt.subplots(figsize=(8,3))
        librosa.display.specshow(mfcc_vis, sr=SAMPLE_RATE, x_axis="time", ax=ax2, cmap="coolwarm")
        st.pyplot(fig2)

# ======================
# History
# ======================
st.subheader("History Prediksi")
if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
    st.dataframe(history_df.tail(10))  # Display the last 10 prediction results
else:
    st.write("Belum ada history prediksi.")
