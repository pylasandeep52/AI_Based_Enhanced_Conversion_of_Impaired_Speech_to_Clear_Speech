import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = load_model("Audio_Classification_Model.h5")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# Function to record audio
def record_audio(duration=3, sr=16000):
    st.info("Recording... Please speak into the microphone.")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording complete!")
    return audio.flatten()

# Function to extract MFCC features
def preprocess_audio(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

# Predict class
def predict_class(audio, sr):
    features = preprocess_audio(audio, sr)
    prediction = model.predict(features)
    class_idx = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    return class_label

# Streamlit app layout
st.title("🎙️ Real-Time Audio Classification App")
st.write("This app records your voice and classifies it using a pre-trained ML model.")

# Parameters
sr = 16000
duration = st.slider("Select Recording Duration (seconds)", 1, 10, 3)

# Button to start recording
if st.button("🎤 Start Recording"):
    audio = record_audio(duration, sr)

    # Optionally play back the audio
    st.audio(audio.tobytes(), format="audio/wav", sample_rate=sr)

    # Predict and display class
    predicted_class = predict_class(audio, sr)
    st.subheader("✅ Predicted Class:")
    st.success(predicted_class)
