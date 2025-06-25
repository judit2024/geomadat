import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = "C:/Users/Judit_2/Documents/Geom adatelemzes beadando/traffic_signs/Train"
img_size = (32, 32)  # Ez lehet 30x30 vagy 64x64 is

data = []
labels = []

# végigmegy az osztályokon (pl. '0', '1', ..., '42')
for class_id in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_id)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            data.append(img)
            labels.append(int(class_id))
        except Exception as e:
            print(f"Hiba a {img_file} fájlnál: {e}")

# numpy tömbbé alakítás
data = np.array(data)
labels = np.array(labels)

# adatok osztása train/test halmazokra
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# normalizálás és one-hot encoding
x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = len(np.unique(labels))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("Adatok sikeresen betöltve és előkészítve.")
