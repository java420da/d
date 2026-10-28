import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dataset
path = r"tiny-imagenet-200/train"

dataset = tf.keras.utils.image_dataset_from_directory(
    path, image_size=(224, 224), batch_size=1, shuffle=False
)

images, captions = [], []

for img, label in dataset.take(50):
    images.append(img.numpy()[0])
    name = dataset.class_names[label.numpy()[0]].replace("_", " ").lower()
    captions.append("<start> a photo of " + name + " <end>")

images = np.array(images)

# 2. Extract image features
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
features = cnn.predict(preprocess_input(images.copy()), verbose=0)

print("Feature shape:", features.shape)

# 3. Tokenize captions
tokenizer = Tokenizer(filters="", lower=True)
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)
vocab = len(tokenizer.word_index) + 1
max_len = max(len(s) for s in sequences)

# 4. Prepare training data
img_data, inputs, targets = [], [], []

for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        img_data.append(features[i])
        inputs.append(seq[:j])
        targets.append(seq[j])

inputs = pad_sequences(inputs, maxlen=max_len - 1, padding="post")
targets = tf.keras.utils.to_categorical(targets, num_classes=vocab)

# 5. Build CNN-LSTM model
image_input = Input(shape=(2048,))
word_input = Input(shape=(max_len - 1,))

a = Dense(128, activation="relu")(image_input)
a = RepeatVector(max_len - 1)(a)
b = Embedding(vocab, 128, mask_zero=True)(word_input)

x = Add()([a, b])
x = LSTM(128)(x)
output = Dense(vocab, activation="softmax")(x)

model = Model([image_input, word_input], output)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 6. Train model
model.fit([np.array(img_data), inputs], targets, epochs=10, batch_size=16)

# 7. Generate caption
def generate_caption(feature):
    caption = "<start>"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_len - 1, padding="post")
        pred = model.predict([feature, seq], verbose=0)[0]
        word = tokenizer.index_word.get(np.argmax(pred), "")

        if not word or word == "<end>":
            break

        caption += " " + word

    return caption.replace("<start>", "").strip()

# 8. Display image and caption
print("Expected caption:", captions[0].replace("<start> ", "").replace(" <end>", ""))
print("Generated caption:", generate_caption(features[0:1]))

plt.imshow(images[0].astype("uint8"))
plt.title(generate_caption(features[0:1]))
plt.axis("off")
plt.show()