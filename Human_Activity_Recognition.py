import numpy as np
import pandas as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_dir = 'path_to_dataset'
file_path = os.path.join(data_dir, 'sensor_data.csv')
data = pd.read_csv(file_path)
data['x'] = (data['x'] - data['x'].mean()) / data['x'].std()
data['y'] = (data['y'] - data['y'].mean()) / data['y'].std()
data['z'] = (data['z'] - data['z'].mean()) / data['z'].std()
def extract_features(df):
    features = []
    labels = []
    for activity in df['activity'].unique():
        activity_data = df[df['activity'] == activity]
        mean_x = activity_data['x'].mean()
        mean_y = activity_data['y'].mean()
        mean_z = activity_data['z'].mean()
        std_x = activity_data['x'].std()
        std_y = activity_data['y'].std()
        std_z = activity_data['z'].std()
        features.append([mean_x, mean_y, mean_z, std_x, std_y, std_z])
        labels.append(activity)
    return np.array(features), np.array(labels)
features, labels = extract_features(data)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_labels)
predictions = knn.predict(test_data)
accuracy = accuracy_score(test_labels, predictions)
print(f'Test accuracy: {accuracy}')
def predict(sensor_readings, model):
    sensor_readings = np.array(sensor_readings).reshape(1, -1)
    prediction = model.predict(sensor_readings)
    return label_encoder.inverse_transform(prediction)
sensor_readings = [mean_x_value, mean_y_value, mean_z_value, std_x_value, std_y_value, std_z_value]
predicted_activity = predict(sensor_readings, knn)
print(f'Predicted activity: {predicted_activity}')
