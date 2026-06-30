import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Loading the dataset
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32

train_dir = "dataset/train"
test_dir = "dataset/test"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    color_mode = "grayscale",
    batch_size = BATCH_SIZE
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    color_mode = "grayscale",
    batch_size = BATCH_SIZE
)

print(train_dataset)
print(test_dataset)

print(train_dataset.class_names)

for images, labels in train_dataset.take(1):
    print(images.shape)
    print(labels.shape)

# Normalisation
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

for images, labels in train_dataset.take(1):
    print(images[0])

for images, labels in train_dataset.take(1):
    print("Minimum:", images.numpy().min())
    print("Maximum:", images.numpy().max())

# Building a CNN model
model = models.Sequential([
    # Added data augumentation for better accuracy
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # First Convotlution Layer
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Second Convotlution Layer
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Third Convotlution Layer
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),

    layers.Dropout(0.5),

    layers.Dense(7, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "models/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[checkpoint, early_stopping]
)

# Save
model.save("models/emotion_recognition_model.keras")

# Test
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")