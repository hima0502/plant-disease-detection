# STEP 1: Install required libraries
!pip install tensorflow keras matplotlib numpy opencv-python kaggle

# STEP 2: Setup Kaggle API to download dataset
import os
from google.colab import files

# Upload your kaggle.json (from Kaggle > Account > Create API Token)
print("👉 Please upload kaggle.json file")
files.upload()

# Move kaggle.json to the right location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# STEP 3: Download PlantVillage dataset
!kaggle datasets download -d emmarex/plantdisease
!unzip -q plantdisease.zip -d dataset

# STEP 4: Organize dataset
# The dataset is already split into folders (train/test) inside 'dataset'
# If needed, you can adjust manually, but let's continue.

# STEP 5: Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training and Validation directories
train_dir = "dataset/PlantVillage"
val_dir = "dataset/PlantVillage"

# STEP 6: Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)   # Split 20% for validation

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(128,128),
                                               batch_size=32,
                                               class_mode='categorical',
                                               subset='training')

val_data = train_datagen.flow_from_directory(val_dir,
                                             target_size=(128,128),
                                             batch_size=32,
                                             class_mode='categorical',
                                             subset='validation')

# STEP 7: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# STEP 8: Train Model
history = model.fit(train_data, validation_data=val_data, epochs=5)

# STEP 9: Save Model
model.save("plant_disease_model.h5")
print("✅ Model trained and saved as plant_disease_model.h5")

# STEP 10: Test on one image
import cv2, numpy as np

# Pick a sample image from validation set
img_path = val_data.filepaths[0]
print("Testing on image:", img_path)

img = cv2.imread(img_path)
img = cv2.resize(img, (128,128))
img = img.astype('float32')/255
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_index = np.argmax(prediction)
class_label = list(train_data.class_indices.keys())[class_index]

print("🌿 Predicted Class:", class_label)
