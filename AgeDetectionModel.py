import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the folders containing the images
emotion_folders = [
    r"C:\Users\Ammar\Downloads\AgeDataset\Dataset\6-20",
    r"C:\Users\Ammar\Downloads\AgeDataset\Dataset\25-30",
    r"C:\Users\Ammar\Downloads\AgeDataset\Dataset\42-48",
    r"C:\Users\Ammar\Downloads\AgeDataset\Dataset\60-98",
]

# Function to load images and labels from the given folders
def load_data(emotion_folders):
    images = []
    labels = []
    for i, folder in enumerate(emotion_folders):
        for filename in os.listdir(folder):
            try:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {filename}")
                    continue
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {os.path.join(folder, filename)}: {e}")
    return np.array(images), np.array(labels)

# Load the data
images, labels = load_data(emotion_folders)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape and normalize the images
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 4 classes: 6-20, 25-30, 42-48, 60-98

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model using the data generator
datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save("age_detection_model.keras")
