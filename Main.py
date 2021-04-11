from Data import Data
from Model import Model
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

    x, y = data.get_training_data()

    num_layers = 3
    hidden_size = 16
    batch_size = 100
    epochs = 100

    recall_list = []
    for train_idx, test_idx in KFold().split(x):

        big_boy = Model(
            num_layers=num_layers,
            hidden_size=hidden_size
        )

        big_boy.train(
            x[train_idx], y[train_idx].ravel(),
            batch_size=batch_size,
            epochs=epochs
        )

        y_pred = big_boy.predict(x[test_idx])

        results = classification_report(y[test_idx], y_pred, output_dict=True)
        recall = results['1.0']['recall']
        recall_list.append(recall)

    recall = np.mean(recall_list)

    print('Average Recall = ' + str(recall))
    with open(f'log.txt', 'a+') as f:
        f.write(f'\n{num_layers},{hidden_size},{batch_size},{epochs},{recall}')

    # Run validation
    big_boy = Model(num_layers=num_layers, hidden_size=hidden_size)
    big_boy.train(x, y, batch_size=batch_size, epochs=epochs)
    x_test, y_test = data.get_test_data()
    y_pred = big_boy.predict(x_test)

    print(f'Neural Network')
    print(classification_report(y_test, y_pred.ravel()))
    print(classification_report(y_test, y_pred.ravel(), output_dict=True)['1.0']['recall'])


if __name__ == '__main__':
    # knn()
    neural_net()
