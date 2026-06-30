import tensorflow as tf

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