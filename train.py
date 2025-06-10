import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
import json

# 1. Load data dan preprocessing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Buat model sederhana fully connected
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# 4. Evaluasi model
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.4f}")

# 5. Simpan model .h5 (optional)
model.save('model/fashion_mnist.h5')

# 6. Simpan arsitektur ke fashion_mnist.json
model_json = model.to_json()
with open('model/fashion_mnist.json', 'w') as json_file:
    json_file.write(model_json)

# 7. Ekstrak weights ke fashion_mnist.npz (format sesuai nn_predict.py: simpan weights dan bias setiap Dense layer)
# Kita simpan ke dictionary dengan key 'W0', 'b0', 'W1', 'b1', ...
weights_dict = {}
dense_layer_index = 0
for layer in model.layers:
    if isinstance(layer, Dense):
        w, b = layer.get_weights()
        weights_dict[f'W{dense_layer_index}'] = w
        weights_dict[f'b{dense_layer_index}'] = b
        dense_layer_index += 1

np.savez('model/fashion_mnist.npz', **weights_dict)

print("Model architecture and weights saved successfully!")
