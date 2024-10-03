import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# Step 1: Data Preparation
# Directory containing images structured as folders of classes
data_dir = r'D:\Projects\Python\Projects\Ingredient_ditection\data\image'  # Update with your dataset path

# Define ImageDataGenerator for Training and Validation sets
batch_size = 32
img_height = 256
img_width = 256

# Data augmentation and rescaling for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.20  # 20% of data used for validation
)

# Training set
train_data = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

# Validation set
val_data = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Step 2: Create the Model
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(25, activation='softmax')  # 25 categories (classes)
])

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Adjust the epochs as needed
)

# Step 5: Evaluate and Print the Accuracy
loss, accuracy = model.evaluate(val_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Step 6: Displaying Classification Report (Optional)
# To see precision, recall, and F1-score for each class
val_labels = val_data.classes
val_pred = np.argmax(model.predict(val_data), axis=-1)

print("Classification Report:\n", classification_report(val_labels, val_pred, target_names=list(train_data.class_indices.keys())))

# Save the trained model (optional)
model.save('models/ingredient_model.h5')
