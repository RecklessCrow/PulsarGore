from Data import Data
from Model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np

from tqdm import tqdm


def knn():
    data = Data()
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
    data = Data()

    # Hyper parameters
    epochs = 750

    # dataset
    x_train, y_train = data.get_training_data()

    # 5-Fold Cross Validation
    recall_list = []
    for train_idx, val_idx in KFold().split(x_train):
        model = Model()
        model.train(epochs, x_train[train_idx], y_train[train_idx])
        recall = model.test(x_train[val_idx], y_train[val_idx])
        recall_list.append(recall)

    model = Model()
    model.train(epochs, x_train, y_train)

    # get results of a trained model on the test set
    x_test, y_test = data.get_test_data()

    print(f'5-Fold Cross Validation Results: Mean Class 1 Recall = {np.mean(recall_list):f}')
    model.test(x_test, y_test, print_report=True)


if __name__ == '__main__':
    # knn()
    neural_net()
