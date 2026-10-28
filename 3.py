import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# 1. Load dataset: 10 images per class
path = "tiny-imagenet-200/train"

dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=64,
    shuffle=True
)

classes = dataset.class_names
num_classes = len(classes)
limit_per_class = 10
total_needed = num_classes * limit_per_class

images, labels = [], []
counts = {i: 0 for i in range(num_classes)}

for x_batch, y_batch in dataset:
    for img, lbl in zip(x_batch.numpy(), y_batch.numpy()):
        if counts[lbl] < limit_per_class:
            images.append(img)
            labels.append(lbl)
            counts[lbl] += 1
    if len(images) >= total_needed:
        break

images = np.array(images, dtype="float32")
y = np.array(labels)

print("Images loaded:", len(images))
print("Classes:", num_classes)

# 2. Extract ResNet50 features in batches
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
base.trainable = False

features = []

for i in range(0, len(images), 16):
    batch = preprocess_input(images[i:i + 16])
    features.append(base.predict(batch, verbose=0))

features = np.concatenate(features, axis=0)
print("Feature shape:", features.shape)

# Release image memory
del images

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train classifier
model = Sequential([
    Input(shape=(2048,)),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=16)

# 5. Evaluate
pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro", zero_division=0))
print("Recall:", recall_score(y_test, pred, average="macro", zero_division=0))
print("F1-score:", f1_score(y_test, pred, average="macro", zero_division=0))

# 6. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, pred,
    cmap="Blues",
    xticks_rotation=90,
    include_values=False
)

plt.title("Confusion Matrix - ResNet50")
plt.tight_layout()
plt.show()

plt.show()
