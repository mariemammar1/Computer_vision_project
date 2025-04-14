import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

base_path = r"C:\Users\Ammar\Downloads\archive\train"

emotion_folders = [
    os.path.join(base_path, "angry"),
    os.path.join(base_path, "disgust"),
    os.path.join(base_path, "fear"),
    os.path.join(base_path, "happy"),
    os.path.join(base_path, "neutral"),
    os.path.join(base_path, "sad"),
    os.path.join(base_path, "surprise"),
]

def load_data(emotion_folders):
    images = []
    labels = []
    for i, folder in enumerate(emotion_folders):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(i)
    return np.array(images), np.array(labels)

images, labels = load_data(emotion_folders)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.layers import Input

model = Sequential()
model.add(Input(shape=(48, 48, 1)))  # Ajoute l'entrée explicite
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
model.save("emotion_detection_model.keras")
