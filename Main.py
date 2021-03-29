from Data import Data
# from Model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np

from tqdm import tqdm


def knn():
    data = Data('data/pulsar_data_train.csv')
    x, y = data.get_training_data()

    max_recall = float('-inf')
    max_k = 0
    max_metric = ''

    for k in tqdm(range(1, 11), total=10, desc='Testing k hyperparameters'):
        for metric in ['euclidean', 'manhattan', 'cosine']:

            classifier = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric,
            )

            recall_list = []

            # train
            for train_idx, test_idx in KFold().split(x):
                classifier.fit(x[train_idx], y[train_idx].ravel())

                y_pred = classifier.predict(x[test_idx])

                results = classification_report(y[test_idx], y_pred, output_dict=True)
                recall = results['1.0']['recall']
                recall_list.append(recall)

            recall = np.mean(recall_list)
            if recall >= max_recall:
                # print("New max recall found!")
                # print(f'\tK={k} Metric={metric}')
                max_recall = recall
                max_k = k
                max_metric = metric

    classifier = KNeighborsClassifier(
        n_neighbors=max_k,
        metric=max_metric,
        n_jobs=-1
    )

    classifier.fit(x, y.ravel())

    x_test, y_test = data.get_test_data()
    print('Optimal hyperparameters:')
    print(f'KNN k={max_k} {max_metric} metric')
    print(classification_report(y_test, classifier.predict(x_test).ravel()))

    print('Actual Recall (positive):', classification_report(y_test, classifier.predict(x_test).ravel(), output_dict=True)['1.0']['recall'])


def neural_net():
    data = Data('data/pulsar_data_train.csv')

    # Create model and train
    big_boy = Model(load_file='models/temp.h5')
    x, y = data.get_training_data()
    # big_boy.train(x, y)

    # Run validation
    x_test, y_test = data.get_test_data()
    y_pred = big_boy.predict(x_test)

    print(f'Neural Network')
    print(classification_report(y_test, y_pred.ravel()))
    print(classification_report(y_test, y_pred.ravel(), output_dict=True)['1.0']['recall'])


if __name__ == '__main__':
    knn()
    # neural_net()
