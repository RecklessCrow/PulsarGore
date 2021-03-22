import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
import numpy as np


def _make_model():
    model = Sequential()

    model.add(Dense(
        units=128,
        activation='relu'
    ))

    model.add(Dropout(0.2))

    model.add(Dense(
        units=1,
        activation='sigmoid'
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    return model


class Model:
    def __init__(self, load_file=None, save_file='models/temp.h5'):
        self.model = _make_model() if load_file is None else load_model(load_file)
        self.save_file = save_file

    def train(self, x, y, epochs=10):

        self.model.fit(
            x,
            y,
            epochs=epochs,
        )

        self.model.save(self.save_file)

    def predict(self, X):
        threshold = 0.40
        predictions = []

        for pred in self.model.predict(X):
            predictions.append([1]) if pred > threshold else predictions.append([0])

        return np.array(predictions)


if __name__ == '__main__':
    my_model = Model('models/temp')