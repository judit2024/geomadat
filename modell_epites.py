import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import cv2

# --- Adatok beolvasása (ha nem pickle-ből mented, akkor újra betöltöd a képeket itt is) ---
data_dir = "C:/Users/Judit_2/Documents/Geom adatelemzes beadando/traffic_signs/Train"
img_size = (32, 32)
data = []
labels = []

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

data = np.array(data) / 255.0
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

num_classes = len(np.unique(labels))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- Modell építése ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Modell tanítása ---
model.fit(x_train, y_train_cat, epochs=20,
          validation_data=(x_test, y_test_cat))

# modell mentése
model.save("modell.h5")
print("Modell sikeresen elmentve.")

# --- Értékelés ---
loss, accuracy = model.evaluate(x_test, y_test_cat)
print(f"Teszt pontosság: {accuracy:.4f}")

history = model.fit(
    x_train, y_train_cat,
    epochs=20,          # 10 kör tanítás (állíthatod)
    batch_size=64,      # egyszerre ennyi adatot dolgoz fel
    validation_split=0.2  # a tanító adat 20%-a validáció lesz
)

# Tanítás után kiértékelés:
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f'Teszt pontosság: {test_acc:.4f}')

# Eredmények vizualizálása


plt.plot(history.history['accuracy'], label='Train pontosság')
plt.plot(history.history['val_accuracy'], label='Validáció pontosság')
plt.xlabel('Epoch')
plt.ylabel('Pontosság')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train veszteség')
plt.plot(history.history['val_loss'], label='Validáció veszteség')
plt.xlabel('Epoch')
plt.ylabel('Veszteség')
plt.legend()
plt.show()
