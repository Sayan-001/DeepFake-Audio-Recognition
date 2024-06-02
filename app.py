import streamlit as st 
import numpy as np
import librosa as lb
import soundfile as sf
import io
import random
import time

from tensorflow.keras.models import load_model

@st.cache()
def init_model():
    model = load_model('./models/DeepFakeDetector_V1.h5')
    return model

def extract_mel_spectrogram(y, sr):
    spectrogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    spectrogram = np.abs(lb.power_to_db(spectrogram, ref=np.max)) / 80
    
    return spectrogram

model = init_model()

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    st.toast("Audio file uploaded successfully!")
    
    data, sr = sf.read(io.BytesIO(audio_bytes))
    y = np.array(data)