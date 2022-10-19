import yaml
import pandas as pd
import numpy as np
from ann import ANN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def get_data():
    """ Function to read data from source
    Args:
        data_path: path to data source
    Returns:
        Dataframe of the breast cancer data
    """
    data_path = "./data/UCI_breast_cancer_data.csv"

    bcwd_data = pd.read_csv(data_path, header=None)

    return bcwd_data


def pre_process(bcwd_data, train_split=0.70, test_split=0.30):
    """ Data Preprocessing for modelling
        - Data Normalising
        - Data Splitting into train and test
    """
    # Data Normalising to change categorical variable to numeric
    target_mapper = {"M": 1, "B": 0}
    bcwd_data[1] = bcwd_data[1].apply(lambda x: target_mapper[str(x)])

    # defining dependant and independant variables
    y = bcwd_data[1]
    X = bcwd_data.drop([1],axis=1)
    y = np.asarray(y)

    # Data Splitting 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_split, 
                                                        train_size=train_split, 
                                                        random_state=30)

    return X_train, X_test, y_train, y_test



def main():
    """ Function to run Neural Network
    Execution sequence:
        get_data 
        pre_process data 
        get_config_data 
        initialization
    """
    data = get_data()
    X_train, X_test, y_train, y_test = pre_process(data)
    print(X_train.shape)
    config = get_config_data()

    sizes = config["sizes"]
    activation_function = config["activation_function"]
    loss_function = config["loss_function"]
    gradient_type = config["gradient_type"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    # create an instance of neural network
    ann = ANN(sizes, activation_function, loss_function, gradient_type)

    # starting Model Training
    y_predicted = ann.train(X_train, y_train, learning_rate, epochs)

    # calculating model accuracy
    y_predicted = (y_predicted>0.5).astype(int)
    model_accuracy = accuracy_score(y_predicted, y_train)
    print("accuracy score:", model_accuracy)



if __name__ == "__main__":
    main()

