import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Concatenate
from tensorflow.keras.models import Model

# 1. Load dataset
path = "tiny-imagenet-200/train"

dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    color_mode="grayscale",
    image_size=(28, 28),
    batch_size=32,
    shuffle=True
)

classes = dataset.class_names
n = len(classes)

images, labels = [], []

for x, y in dataset:
    images.extend(x.numpy() / 255.0)
    labels.extend(y.numpy())

images = np.array(images, dtype="float32")
labels = tf.keras.utils.to_categorical(labels, n)

print("Classes:", classes)
print("Images:", images.shape)

# 2. Generator
noise = Input(shape=(100,))
text = Input(shape=(n,))

x = Concatenate()([noise, text])
x = Dense(128, activation="relu")(x)
x = Dense(784, activation="sigmoid")(x)
output = Reshape((28, 28, 1))(x)

G = Model([noise, text], output)

# 3. Discriminator
img = Input(shape=(28, 28, 1))
label = Input(shape=(n,))

x = Flatten()(img)
x = Concatenate()([x, label])
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

D = Model([img, label], output)
D.compile(optimizer="adam", loss="binary_crossentropy")

# 4. Combined GAN
D.trainable = False

z = Input(shape=(100,))
t = Input(shape=(n,))

fake = G([z, t])
valid = D([fake, t])

GAN = Model([z, t], valid)
GAN.compile(optimizer="adam", loss="binary_crossentropy")

# 5. Training
D.trainable = True
batch = min(32, len(images))

for epoch in range(100):
    idx = np.random.randint(0, len(images), batch)
    real_img = images[idx]
    real_labels = labels[idx]

    noise = np.random.randn(batch, 100).astype("float32")
    fake_img = G.predict([noise, real_labels], verbose=0)

    D.train_on_batch([real_img, real_labels], np.ones((batch, 1)))
    D.train_on_batch([fake_img, real_labels], np.zeros((batch, 1)))

    D.trainable = False
    noise = np.random.randn(batch, 100).astype("float32")
    GAN.train_on_batch([noise, real_labels], np.ones((batch, 1)))
    D.trainable = True

    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1)

# 6. Generate image for any selected class
class_name = classes[-1]
class_id = classes.index(class_name)

text = tf.keras.utils.to_categorical([class_id], n)
noise = np.random.randn(1, 100).astype("float32")

generated = G.predict([noise, text], verbose=0)

plt.imshow(generated[0, :, :, 0], cmap="gray")
plt.title("Generated: " + class_name)
plt.axis("off")
plt.show()