import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_network.network import NeuralNetwork
from neural_network.gradient_descent import batch_gradient_descent, stochastic_gradient_descent


def get_config_data():
    """ Read the configurations for Neural Network
    Returns:
        sizes: Number of layers and number of neurons in the network
        learning_rate
        activation_function: list of activation functions from layer to another - Sigmoid,tanh,ReLU,LeakyReLU
        loss_function: MSE(Mean square error), BCE(Binary Cross Entropy), CE(Cross Entropy)
        gradient_type: FGD(full gradient descent) MB(mini-batch gradient descent)
        epochs: number of epochs
    """
    config_path = "./config.yml"

    with open(config_path) as f:
        configurations = yaml.safe_load(f)

    return configurations


def load_dataset(path):
    """ Function to read data from source
    Args:
        path: path to data source
    Returns:
        Dataframe of the breast cancer data
    """

    data = pd.read_csv(path)

    return data


def pre_process_data(data, train_split=0.70, val_split=0.30, test_split=0.50):
    """ Data Preprocessing for modelling
        - Data Normalising
        - Data Splitting into train and test
    """
    # drop unamed column
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

    X = data.drop('diagnosis', axis=1)

    y = data.diagnosis
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Data Splitting
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                        test_size=val_split,
                                                        train_size=train_split,
                                                        random_state=30)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split, random_state=30)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    X_val = sc.fit_transform(X_val)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    y_val = y_val.reshape(len(y_val), 1)

    return X_train, X_test, X_val, y_train, y_test, y_val


def predict(model, test_input, test_output):
    predictions = model.feed_forward(test_input)
    prediction_accuracy = accuracy_score((predictions > 0.5).astype(int), test_output)
    return prediction_accuracy


def main(config):
    
    # unpacking the configurations
    sizes = config["sizes"]
    activation_functions = config["activation_function"]
    loss_function = config["loss_function"]
    gradient_type = config["gradient_type"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    batch_size = 100
    data_path = "./data/data.csv"

    # loading the data
    data = load_dataset(data_path)

    # getting the data splits
    X_train, X_test, X_val, y_train, y_test, y_val = pre_process_data(data)

    # starting the model training
    layers = []
    for size, activation_function in zip(sizes, activation_functions):
        layers.append((size, activation_function))

    model = NeuralNetwork(X_train.shape[1],
                          layers,
                          learning_rate,
                          loss_function
                          ).create_network()

    print(f"Model architecture...{model.architecture}")

    if gradient_type == "SGD":
        result = stochastic_gradient_descent(epochs,
                                             model,
                                             X_train,
                                             y_train,
                                             X_val,
                                             y_val,
                                             batch_size)
        
    else:
        result = batch_gradient_descent(epochs,
                                        model,
                                        X_train,
                                        y_train,
                                        X_val,
                                        y_val
                                        )

    # model training metrics
    train_accuracy = result["accuracy"]["train"]
    train_loss_list = result["losses"]["train"]
    train_loss = sum(train_loss_list)/len(train_loss_list)

    # model validation metrics
    validation_accuracy = result["accuracy"]["validation"]
    validation_loss_list = result["losses"]["validation"]
    validation_loss = sum(validation_loss_list)/len(validation_loss_list)

    # model testing metrics
    test_accuracy = predict(model, X_test, y_test)

    print(f"Model accuracy score...:{train_accuracy}")
    print("Making prediction...")
    print(f"Model prediction accuracy score...:{test_accuracy}")

    return train_accuracy, train_loss_list, train_loss,\
            validation_accuracy, validation_loss_list, validation_loss,\
            test_accuracy


if __name__ == '__main__':
    config = get_config_data()
    main(config)
