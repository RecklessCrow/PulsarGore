import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

# Fix tf GPU parameters on my PC
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def _make_model():
    model = Sequential()

    model.add(Dense(
        units=32,
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

        return np.array([[1] if pred > threshold else [0] for pred in self.model.predict(X)])


if __name__ == '__main__':
    my_model = Model('models/temp')
