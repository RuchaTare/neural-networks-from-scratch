import yaml
import pandas as pd
import numpy as np
from ann import ANN
from sklearn.model_selection import train_test_split


def get_config_data():
    """ Read the configurations for Neural Network
    """
    config_path = "./config.yml"

    with open(config_path) as f:
        configurations = yaml.safe_load(f)
    
    return configurations


def get_data():
    """ Function to read data from source
    Args:
        data_path: path to data source
    """
    data_path = "./data/UCI_breast_cancer_data.csv"

    bcwd_data = pd.read_csv(data_path, header=None)

    return bcwd_data


def pre_process(bcwd_data, train_split=0.70, test_split=0.30):
    """ Data Preprocessing for modelling
        - Data Normalising
        - Data Splitting
    """
    # Data Normalising
    target_mapper = {"M": 1, "B": 0}
    bcwd_data[1] = bcwd_data[1].apply(lambda x: target_mapper[str(x)])

    #defining dependant and independant variables
    y = bcwd_data[1]
    X = bcwd_data.drop([1],axis=1)

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_split, 
                                                        train_size=train_split, 
                                                        random_state=30)

    return X_train, X_test, y_train, y_test


def main():
    """ Function to run Neural Network
    """
    data = get_data()
    X_train, X_test, y_train, y_test = pre_process(data)
    print(X_train.head())
    config = get_config_data()

    sizes = config["sizes"]
    activation_function = config["activation_function"]
    loss_function = config["loss_function"]
    gradient_type = config["gradient_type"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    # create an instance of Neural Network
    ann = ANN(sizes, activation_function, loss_function, gradient_type)

    ann.train(X_train, y_train, learning_rate, epochs)


if __name__ == "__main__":
    main()

