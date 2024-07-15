import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
def load_data(data_dir):
    data = []
    labels = []
    emotion_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for emotion in emotion_map:
        emotion_dir = os.path.join(data_dir, emotion)
        for img in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            data.append(img)
            labels.append(emotion_map[emotion])
    data = np.array(data).reshape(-1, 48, 48, 1) / 255.0
    labels = to_categorical(labels, num_classes=7)
    return data, labels
data_dir = 'path_to_dataset'
data, labels = load_data(data_dir)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = create_model()
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
model.fit(datagen.flow(train_data, train_labels, batch_size=64),
          validation_data=(test_data, test_labels),
          epochs=50)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')
model.save('emotion_recognition_model.h5')
from keras.models import load_model
model = load_model('emotion_recognition_model.h5')
def predict_emotion(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    predictions = model.predict(img)
    emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    return emotion_map[np.argmax(predictions)]
image_path = 'path_to_test_image.jpg'
predicted_emotion = predict_emotion(image_path, model)
print(f'Predicted emotion: {predicted_emotion}')
