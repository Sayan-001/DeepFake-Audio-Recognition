import streamlit as st 
import numpy as np
import librosa as lb
import soundfile as sf
import io
import time

from tensorflow.keras.models import load_model

st.set_page_config(page_title="DeepFake Audio Recognization", page_icon="🎵", layout="wide")

model = load_model('./models/DeepFakeDetector_V1.h5')

def load_audio(audio_bytes):
    data, _ = sf.read(io.BytesIO(audio_bytes))
    y = np.array(data)
    y = y[:, 0] if y.ndim > 1 else y
    
    return y

def extract_mel_spectrogram(y):
    spectrogram = np.abs(lb.power_to_db(lb.feature.melspectrogram(y=y, n_mels=64), ref=np.max)) / 80
    return spectrogram

def create_dataset(y):
    sr = 22050
    duration = 1.5
    intervals = range(0, len(y), int(sr*duration))
    
    try:
        X = np.zeros((len(intervals)-1, 64, 65), dtype=np.float32)
    except ValueError:
        return None
    
    for i in range(len(intervals)-1):
        audio = y[intervals[i]:intervals[i+1]]
        X[i] = extract_mel_spectrogram(audio)
        
    return X

st.title("DeepFake Audio Recognization")

col1, _, col2 = st.columns((5, 1, 4))

with col1:
    audio_file = st.file_uploader("Upload an audio file (Duration must be atleast 1.5 seconds):", type=["wav", "mp3", "m4a"])
    
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        st.toast("Audio file uploaded successfully!")
        
        y = load_audio(audio_bytes)
        
        analyzse = st.button("Analyze")
        
        if analyzse:
            start = time.perf_counter()
            
            with st.spinner("Extracting data..."):
                X = create_dataset(y)
                if X is None:
                    st.error("Audio duration must be atleast 1.5 seconds!")
                    st.stop()
                
            with st.spinner("Predicting..."):
                predictions = model.predict(X)
                
            positive = len(predictions[predictions > 0.5])
            negative = len(predictions[predictions <= 0.5])
                
            pred_mean = np.mean(predictions)
            prediction = "Fake" if pred_mean > 0.5 else "Real"
            end = time.perf_counter()
            
            st.markdown("---")
            st.markdown("**Analysis Results**")
            col11, col12, col13 = st.columns((1, 1, 1))
            with col11:
                st.metric("Fake Strength", f"{pred_mean: .3f}")
            with col12:
                st.metric("Real Strength", f"{1-pred_mean: .3f}")
            with col13:
                st.metric("Prediction", prediction)
            
            st.success("Analysis completed!")
            
with col2:
    st.image(r".\images\audio.jpg")
    st.markdown("**Description**")
    st.write("""This is a simple DeepFake Audio Recognization app. It uses a 
                Convolutional Neural Network to predict whether the audio is real or fake. 
                The model was trained on a Kaggle dataset of real and fake audio files. 
                The model has an test accuracy of ~95%.""")
    st.markdown("**How it works**")
    st.write("""The model accepts a mel spectrogram sample of 64 mels and duration of 1.5 seconds,
                which corresponds to a 64x65 matrix. The model then predicts the probability of the
                audio being fake. If the probability is greater than 0.5, the model predicts the audio
                as fake. Otherwise, it predicts the audio as real.""")
    st.write("Total model parameters: ~1.8M")
    st.markdown("**Limitations**")
    st.write("""However, the model is only capable to the extent of the data it was trained on.
                It is not perfect and may not predict correctly to unforeseen data on the internet.
                It is still a work in progress and will be improved in the future.""")