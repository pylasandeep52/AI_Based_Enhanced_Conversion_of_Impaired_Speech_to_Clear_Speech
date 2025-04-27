import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model
model = load_model('Audio_Classification_Model.h5')

# Load the LabelEncoder used during training
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Capture dynamic audio
def record_audio(duration=3, sr=16000):
    """
    Record audio from the microphone for a given duration.
    Args:
        duration (int): Duration of the recording in seconds.
        sr (int): Sampling rate.
    Returns:
        np.ndarray: Recorded audio as a NumPy array.
    """
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")
    return audio.flatten()

# Pre-process the audio
def preprocess_audio(audio, sr):
    """
    Extract features from the audio (e.g., MFCCs).
    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate.
    Returns:
        np.ndarray: Extracted features.
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)  # Reshape for model input

# Predict the class of the audio
def predict_class(audio, sr=16000):
    """
    Predict the class of the given audio sample.
    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate.
    Returns:
        str: Predicted class label.
    """
    # Pre-process the audio
    features = preprocess_audio(audio, sr)
    # Predict the class
    prediction = model.predict(features)
    class_idx = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    return class_label

# Main function for dynamic classification
if __name__ == "__main__":
    # Record audio for 3 seconds
    sr = 16000  # Sampling rate
    duration = 5  # Duration in seconds
    audio = record_audio(duration, sr)

    # Predict the class of the recorded audio
    predicted_class = predict_class(audio, sr)
    print(f"Predicted Class: {predicted_class}")