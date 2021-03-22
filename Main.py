from Data import Data
from Model import Model


def main():

    data = Data('data/pulsar_data_train.csv')
    x, y = data.get_training_data()
    test_x, test_y = data.get_test_data()

    big_boy = Model(load_file='models/temp.h5')
    big_boy.train(x, y)
    Y_pred = big_boy.predict(test_x)

    data.validate(Y_pred)


if __name__ == '__main__':
    main()
