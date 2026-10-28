import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import (
    ResNet50, preprocess_input, decode_predictions
)
from tensorflow.keras.models import Model

# 1. Load image
path = r"tiny-imagenet-200/test/images/test_0.JPEG"

img = tf.keras.utils.load_img(path, target_size=(224, 224))
arr = tf.keras.utils.img_to_array(img)
x = preprocess_input(np.expand_dims(arr, axis=0))

print("Original image shape:", arr.shape)
print("Preprocessed image shape:", x.shape)

# 2. Load pretrained ResNet50
model = ResNet50(weights="imagenet")

# 3. Predict top-5 classes
pred = model.predict(x, verbose=0)
top5 = decode_predictions(pred, top=5)[0]

print("\nPrediction output shape:", pred.shape)
print("Top-5 predictions:")

for _, label, score in top5:
    print(label, ":", round(float(score) * 100, 2), "%")

print("Top-1 prediction:", top5[0][1])

# 4. Compare actual and predicted labels
actual = "golden retriever"

print("Actual class:", actual)
print("Comparison:",
      "MATCH" if actual in top5[0][1].lower() else "NO MATCH")

# 5. Display original image
plt.imshow(img)
plt.title("Actual: Golden Retriever\nPredicted: " + top5[0][1])
plt.axis("off")
plt.show()

# 6. Visualize intermediate feature maps
layers = ["conv1_relu", "conv2_block3_out", "conv5_block3_out"]

feature_model = Model(
    model.input,
    [model.get_layer(name).output for name in layers]
)

features = feature_model.predict(x, verbose=0)

for name, fmap in zip(layers, features):
    print(name, "feature-map shape:", fmap.shape)

    plt.figure(figsize=(8, 4))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(fmap[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(name + " - Feature Maps")
    plt.tight_layout()
    plt.show()