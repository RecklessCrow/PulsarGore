import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

# Fix tf GPU parameters on my PC

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def _make_model(num_layers, hidden_size):
    model = Sequential()

    for _ in range(num_layers):
        model.add(Dense(
            units=hidden_size,
            activation='tanh'
        ))

    model.add(Dense(
        units=1,
        activation='sigmoid'
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.Recall()]
    )

    return model


class Model:
    def __init__(self, load_file=None, save_file='models/temp.h5', num_layers=1, hidden_size=64):
        self.model = _make_model(num_layers=1, hidden_size=64) if load_file is None else load_model(load_file)
        self.save_file = save_file

    def train(self, x, y, epochs=40, batch_size=50, validation=False):

        validation_split = 0.2 if validation else None

        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=2
        )

        self.model.save(self.save_file)

    def predict(self, X):
        pred = self.model.predict(X)
        return np.array((pred > 0.5), dtype=float)


if __name__ == '__main__':
    my_model = Model('models/temp')
