import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import librosa.display

# Load model & encoder
model = load_model('Audio_Classification_Model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# UI Setup
st.set_page_config(page_title="🎙️ Audio Classifier", layout="centered")
st.title("🎤 Real-Time Audio Classification")
st.markdown("Click the button below to record your voice and get a predicted class.")

# Record function
def record_audio(duration=10, sr=16000):
    st.info("⏺️ Recording... Please speak into the microphone.")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording complete!")
    return audio.flatten(), sr

# Preprocessing
def preprocess_audio(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1), mfccs

# Predict function
def predict_class(features):
    prediction = model.predict(features)
    class_idx = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    return class_label

# Main UI Button
if st.button("🎙️ Record Now"):
    audio, sr = record_audio()
    features, mfcc = preprocess_audio(audio, sr)
    predicted_class = predict_class(features)

    # Save & play recorded audio
    sf.write("temp.wav", audio, sr)
    st.audio("temp.wav", format='audio/wav')

    # Show predicted class
    st.success(f"✅ **Predicted Class:** `{predicted_class}`")

    # Optional: MFCC Visualization
    st.markdown("#### 🎵 MFCC Feature Visualization")
    fig, ax = plt.subplots()
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    st.pyplot(fig)
