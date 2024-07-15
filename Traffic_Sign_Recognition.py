import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
def load_data(data_dir):
    data = []
    labels = []
    classes = len(os.listdir(data_dir))
    for i in range(classes):
        path = os.path.join(data_dir, str(i))
        images = os.listdir(path)
        for img in images:
            try:
                image = cv2.imread(os.path.join(path, img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (32, 32))
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(e)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
data_dir = 'path_to_dataset'
data, labels = load_data(data_dir)
data = data / 255.0
labels = to_categorical(labels, 43)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
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
history = model.fit(datagen.flow(train_data, train_labels, batch_size=64),
                    validation_data=(test_data, test_labels),
                    epochs=30)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()
model.save('traffic_sign_recognition_model.h5')
from keras.models import load_model
model = load_model('traffic_sign_recognition_model.h5')
def predict_traffic_sign(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 3) / 255.0
    predictions = model.predict(img)
    return np.argmax(predictions)
image_path = 'path_to_test_image.jpg'
predicted_class = predict_traffic_sign(image_path, model)
print(f'Predicted class: {predicted_class}')
