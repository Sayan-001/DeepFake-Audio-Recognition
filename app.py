import streamlit as st 
import numpy as np
import librosa as lb
import soundfile as sf
import io
import time

from tensorflow.keras.models import load_model

def init_model():
    m = load_model('./models/DeepFakeDetector_V1.h5')
    return m

model = init_model()

def extract_mel_spectrogram(y):
    spectrogram = np.abs(lb.power_to_db(lb.feature.melspectrogram(y=y, n_mels=64), ref=np.max)) / 80
    return spectrogram

def create_dataset(y):
    sr = 22050
    intervals = range(0, len(y), int(sr*1.5))
    X = np.zeros((len(intervals)-1, 64, 65), dtype=np.float32)
    
    for i in range(len(intervals)-1):
        audio = y[intervals[i]:intervals[i+1]]
        X[i] = extract_mel_spectrogram(audio)
        
    return X
        
        

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    st.toast("Audio file uploaded successfully!")
    
    data, _ = sf.read(io.BytesIO(audio_bytes))
    y = np.array(data)
    y = y[:, 0] if y.ndim > 1 else y
    
    analyzse = st.button("Analyze")
    
    if analyzse:
        with st.spinner("Creating data..."):
            X = create_dataset(y)
            time.sleep(1)
            
        with st.spinner("Predicting..."):
            predictions = model.predict(X)
            time.sleep(1)
            
        pred_mean = np.mean(predictions)
        
        st.write(f"Prediction: {pred_mean: .3f}")
        st.write("Fake" if pred_mean > 0.5 else "Real")
            