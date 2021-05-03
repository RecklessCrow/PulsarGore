from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, save_model, load_model

# Fix tf GPU parameters on my PC
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

time_stamp = datetime.now().strftime('%m-%d_%H-%M-%S')


class Model:
    def __init__(self, load_file=None, save_file=None):
        self.model = self.__make_model()

        if load_file is None:
            self.load(load_file)

        if save_file is None:
            self.save_file = f'../models/{time_stamp}'

    @staticmethod
    def __make_model():
        """
        Makes a simple classification model
        :return: Keras Model
        """

        dropout = 0.5

        model = Sequential([

            Dense(
                units=64,
                activation='gelu',
                kernel_initializer='he_normal'
            ),
            Dropout(dropout),
            BatchNormalization(),

            Dense(
                units=32,
                activation='gelu',
                kernel_initializer='he_normal'
            ),
            Dropout(dropout),
            BatchNormalization(),

            Dense(
                units=32,
                activation='gelu',
                kernel_initializer='he_normal'
            ),
            Dropout(dropout),
            BatchNormalization(),

            Dense(
                2,
                activation='softmax'
            )

        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=[tf.keras.metrics.Recall(class_id=1)],  # tracks recall for positive class
        )

        return model

    def train(self, epochs, x, y):
        """
        Trains a model on test data
        :param epochs: Number of times to iterate over the full dataset
        :param x: Data to train with
        :param y: Labels of data
        """

        # creates a dict of class weights to weight loss as if the classes were balanced
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y.argmax(axis=1)
        )
        weights = dict(enumerate(class_weights))

        self.model.fit(
            x, y,
            steps_per_epoch=1,
            epochs=epochs,
            class_weight=weights
        )

    def test(self, x, y_true, print_report=False):
        """
        Compare a models predictions to actual labels
        :param x: data
        :param y_true: real labels
        :param print_report: wheter to print a report of
        :return: Class 1 recall
        """
        y_pred = self.predict(x)
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.argmax(axis=1)

        if print_report:
            print(classification_report(y_true, y_pred, target_names=['noise', 'pulsar']))

        results = classification_report(y_true, y_pred, output_dict=True)
        recall = results['1']['recall']

        return recall

    def predict(self, X):
        return self.model.predict(X)

    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        save_model(filename, self.model)


if __name__ == '__main__':
    my_model = Model('models/temp')
