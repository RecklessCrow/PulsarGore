import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


class Data:
    def __init__(self):
        self.label_encoder = OneHotEncoder(sparse=False)
        self.imputer = KNNImputer()
        self.scaler = RobustScaler()

        train = pd.read_csv('data/train_data.csv')
        self.X_train = train.iloc[:, :-1]
        self.X_train = self.imputer.fit_transform(self.X_train)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.Y_train = np.array(train['target_class']).reshape(-1, 1)
        self.Y_train = self.label_encoder.fit_transform(self.Y_train)

        test = pd.read_csv('data/test_data.csv')
        self.X_test = test.iloc[:, :-1]
        self.X_test = self.imputer.transform(self.X_test)
        self.X_test = self.scaler.transform(self.X_test)
        self.Y_test = np.array(test['target_class']).reshape(-1, 1)
        self.Y_test = self.label_encoder.transform(self.Y_test)

    def get_training_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test


if __name__ == '__main__':
    data = Data()
    print(data.get_training_data())
