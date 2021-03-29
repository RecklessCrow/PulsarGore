import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self, file_name, seed=1):
        df = pd.read_csv(file_name)
        X = df.iloc[:, :-1]
        Y = np.array(df.iloc[:, -1]).reshape(-1, 1)

        X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    def get_training_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test
    
    def scale(self, X):
        return self.scaler.transform(X)


if __name__ == '__main__':
    data = Data('data/pulsar_data_train.csv')
    print(data.get_training_data())
