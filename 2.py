import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# 1. Load image
img = tf.keras.utils.load_img(
    r"tiny-imagenet-200/test/images/test_0.JPEG",
    target_size=(224, 224)
)

arr = tf.keras.utils.img_to_array(img)
x = preprocess_input(np.expand_dims(arr, axis=0))

# 2. Load ResNet50
model = ResNet50(weights="imagenet")

# 3. Select three layers
layers = ["conv1_relu", "conv2_block3_out", "conv5_block3_out"]

feature_model = Model(
    model.input,
    [model.get_layer(name).output for name in layers]
)

# 4. Extract and display feature maps
features = feature_model.predict(x, verbose=0)

for name, fmap in zip(layers, features):
    print(name, "Feature map shape:", fmap.shape)

    plt.figure(figsize=(8, 4))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(fmap[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(name + " - Feature Maps")
    plt.tight_layout()
    plt.show()

# 5. Visualize convolutional filters
weights, _ = model.get_layer("conv1_conv").get_weights()
print("Filter shape:", weights.shape)

plt.figure(figsize=(8, 4))

for i in range(8):
    filt = weights[:, :, :, i]
    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)

    plt.subplot(2, 4, i + 1)
    plt.imshow(filt)
    plt.axis("off")

plt.suptitle("ResNet50 - Convolutional Filters")
plt.tight_layout()
plt.show()

# 6. Display original image
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()