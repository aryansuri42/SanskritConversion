import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Reshape,
                                      LSTM, TimeDistributed, Dropout)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT = 64
IMG_WIDTH = 256
NUM_CLASSES = 128
MAX_SEQ_LENGTH = 50

def build_cnn(input_shape):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dropout(0.3))
    return cnn_model

input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
cnn_model = build_cnn(input_shape)
cnn_output_shape = cnn_model.output_shape[-1]
cnn_features = Reshape((MAX_SEQ_LENGTH, cnn_output_shape // MAX_SEQ_LENGTH))(cnn_model.output)
lstm_layer = LSTM(256, return_sequences=True)(cnn_features)
lstm_layer = LSTM(256, return_sequences=True)(lstm_layer)
dense_output = TimeDistributed(Dense(NUM_CLASSES, activation='softmax'))(lstm_layer)
model = Model(inputs=cnn_model.input, outputs=dense_output)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

sanskrit_characters = 'अआइईउऊऋॠएऐओऔकखगघङचछजझञ...'
char_to_int = {char: idx for idx, char in enumerate(sanskrit_characters)}

def text_to_sequence(text):
    return [char_to_int[char] for char in text if char in char_to_int]

def pad_sequence(seq):
    return pad_sequences([seq], maxlen=MAX_SEQ_LENGTH, padding='post')[0]

def load_data(image_paths, labels):
    images, sequences = [], []
    for img_path, label in zip(image_paths, labels):
        img = preprocess_image(img_path)
        seq = text_to_sequence(label)
        seq = pad_sequence(seq)
        images.append(img)
        sequences.append(seq)
    return np.array(images), np.array(sequences)

def predict_text(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    pred_classes = np.argmax(preds, axis=-1)
    int_to_char = {v: k for k, v in char_to_int.items()}
    predicted_text = ''.join([int_to_char[idx] for idx in pred_classes[0] if idx in int_to_char])
    return predicted_text
