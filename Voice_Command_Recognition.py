import os
import wave
import numpy as np
import contextlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define categories
categories = ['start', 'stop', 'left', 'right']

# Directory containing audio samples
data_dir = 'path_to_audio_samples'

# Load and preprocess audio data
def load_audio_data(data_dir, categories):
    data = []
    labels = []

    for idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for audio_file in os.listdir(category_dir):
            try:
                file_path = os.path.join(category_dir, audio_file)
                with contextlib.closing(wave.open(file_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    signal = f.readframes(frames)
                    signal = np.frombuffer(signal, dtype=np.int16)
                    signal = signal[:rate]  # Use only the first second of audio for consistency
                    data.append(signal)
                    labels.append(idx)
            except Exception as e:
                print(f"Error loading audio file: {e}")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

data, labels = load_audio_data(data_dir, categories)

# Split data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define and train k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
train_data_flat = train_data.reshape(len(train_data), -1).astype(np.float32)
knn.fit(train_data_flat, train_labels)

# Evaluate the model
test_data_flat = test_data.reshape(len(test_data), -1).astype(np.float32)
predictions = knn.predict(test_data_flat)
accuracy = accuracy_score(test_labels, predictions)
print(f'Test accuracy: {accuracy}')

# Predict function
def predict(audio_path, model):
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        signal = f.readframes(frames)
        signal = np.frombuffer(signal, dtype=np.int16)
        signal = signal[:rate]  # Use only the first second of audio for consistency
        signal = signal.reshape(1, -1).astype(np.float32)
        prediction = model.predict(signal)
        return categories[prediction[0]]

# Example usage
audio_path = 'path_to_test_audio.wav'
predicted_command = predict(audio_path, knn)
print(f'Predicted command: {predicted_command}')
