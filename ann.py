import numpy as np

class ANN:

    def __init__(self, sizes, activation_function, loss_function, gradient_type):
        """ Initializing the input variables
        Args:
            sizes:
            learning_rate:
            activation_function:
            loss_function:
            gradient_type:
            epochs:
        """
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.gradient_type = gradient_type

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = []

        # Taking Xaviers Weight initialization
        for i in range(1, self.num_of_layers):
            self.weights.append(np.sqrt(2/sizes[i-1]) * np.random.randn(sizes[i-1], sizes[i]))

        # Defining numpy zeroes for weight and bias list for updates in back propogation
        delta_weight = []
        delta_bias = []

        for i in range(1, self.num_of_layers):
            delta_weight.append(np.zeros((sizes[i], sizes[i-1])))
            delta_bias.append(np.zeros((sizes[i], sizes[i-1])))
        
        self.delta_weight = delta_weight
        self.delta_bias = delta_bias
        
        layer_outputs=[]
        for i in range(self.num_of_layers):
            a = np.zeros(sizes[i])
            layer_outputs.append(a)
        self.layer_outputs = layer_outputs

    ######################################################
    #    Define all activation functions and its prime   #     
    ######################################################
    def sigmoid(self, z):
        """ Define Sigmoid Activiation Function
        """
        return 1.0/(1.0+np.exp(-z))


    def sigmoid_prime(self, z):
        """ Define Sigmoid Activation Function Prime
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def tanh(self, z):
        """ Define tanh Activation Function
        """
        return np.tanh(z)


    def tanh_prime(self, z):
        """ Define tanh Activation Function prime
        """
        return 1 - np.tanh(z) ** 2


    def ReLU(self, z):
        """ Define ReLU Activation Function
        """
        return np.maximum(0, z)

    
    def ReLU_prime(self, z):
        """ Define ReLU Activation Function prime
        """ 
        print("Under ReLu =============================", z)
        return (z > 0) * 1


    def leaky_ReLU(z):
        """ Define Leaky ReLU activation function
        """
        if z < 0:
            return z*0.01
        else:
            return z


    def leaky_ReLU_prime(z):
        """ Define Leaky ReLU Activation Function prime
        """
        if z < 0:
            return 0.01
        else:
            return 1


    def MSE(self, predicted, actual, derivative =False):
        """ Define MSE Cost Function
        Args:
            output: y-hat (predicted value)
            input: y (actual value)
        """
        if derivative:
            loss_derivative = 2*(predicted-actual).mean()
            return loss_derivative
        else:
            loss_score = np.square(predicted-actual).mean()
            return loss_score


    def binary_cross_entropy(self, predicted, actual, derivative=False):
        """ Define Binary Cross entropy function
        Args:
            predicted: y-hat (predicted value)
            actual: y (actual value)
        """
        if derivative:
            loss_derivative = -(actual/predicted - (1-actual)/(1-predicted))
            return loss_derivative
        else:
            loss_score = -np.mean((actual * np.log(predicted)) + ((1-actual)*np.log(1-predicted)))
            return loss_score
    

    # requires changes
    def cross_entropy(self, predicted, actual, derivative=False):
        """ Define root mean squared error 
        Args:
            predicted: y-hat (predicted value)
            actual: y (actual value)
        """
        if derivative:
            loss_derivative = -np.sum(actual * np.log(predicted))
            return loss_derivative
        else:
            loss_score = -np.sum(actual * np.log(predicted))
            return loss_score


    def calculate_loss(self, actual, predicted, loss_function):
        """ Check for the loss function to be used
            Calculate and return the loss_score and derivate of loss_score
        Args:
            actual:
            predicted:
            loss_function:
        """
        if loss_function == "MSE":
            loss_score = self.MSE(predicted, actual, derivative=False)
            loss_derivative = self.MSE(predicted, actual, derivative=True)
            
            return loss_score, loss_derivative
        
        if loss_function == "BCE":
            loss_score = self.binary_cross_entropy(predicted, actual, derivative=False)
            loss_derivative = self.binary_cross_entropy(predicted, actual, derivative=True)
            
            return loss_score, loss_derivative
        
        if loss_function == "CE":
            loss_score = self.cross_entropy(predicted, actual, derivative=False)
            loss_derivative = self.cross_entropy(predicted, actual, derivative=True)
            
            return loss_score, loss_derivative


    def feed_forward(self, input):
        """ Define Feed Forward Pass
        """
        # pass our inputs through our neural network
        activation_index = 0
        layer_output = input
        self.layer_outputs[0] = layer_output
        print("===========================In Feed forward=========================================")

        for b, w in zip(self.biases, self.weights):
            print("Feed Forward Shape", layer_output.shape)
            z = (np.dot(layer_output, w) +b)
            if self.activation_function[activation_index] == 'sigmoid':
                layer_output = self.sigmoid(z)
            if  self.activation_function[activation_index] == 'tanh':
                layer_output = self.tanh(z)
            if  self.activation_function[activation_index] == 'ReLU':
                layer_output = self.ReLU(z)
            if  self.activation_function[activation_index] == 'leaky_ReLU':
                layer_output = self.leaky_ReLU(z)

            self.layer_outputs[activation_index+1] = layer_output
            print("Feed Forward Shape", layer_output.shape)
            activation_index+=1
            
        return layer_output


    def back_propogation(self, loss_derivative, output):
        """ Define Back Propogation
        Args:
            loss_derivative: 
            output: 
        """
        print("Old Output Shape", output.shape)

        
        for i in reversed(range(self.num_of_layers-1)):
            output = self.layer_outputs[i+1]
            print("New Output Shape", output.shape)
            if self.activation_function[i] == 'sigmoid':
                activation_prime = self.sigmoid_prime(output)
            if  self.activation_function[i] == 'tanh':
                activation_prime = self.tanh_prime(output)
            if  self.activation_function[i] == 'ReLU':
                activation_prime = self.ReLU_prime(output)
            if  self.activation_function[i] == 'leaky_ReLU':
                activation_prime = self.leaky_ReLU_prime(output)
            
            # 
            delta = loss_derivative * activation_prime

            # Getting a transpose of 2D array for activation_prime
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # Making a 2D array of the output 
            current_layer_output = self.layer_outputs[i]
            output_reshaped = current_layer_output.reshape(current_layer_output.shape[0], -1)

            # 
            self.delta_weight[i] = np.dot(output_reshaped, delta_reshaped)

            #
            self.delta_bias[i] = delta

            # 
            print("Weights", self.weights[i].shape)
            loss_derivative = np.dot(delta, self.weights[i].T)
    

    def full_gradient_descent(self, learning_rate):
        """ Minimise the gradient
        Args: 
            delta_weight:
            delta_bias:
            learning_rate:
        """
        for i in range(self.num_of_layers):
            print ("========================= IN FGD========================")
            weight = self.weights[i]
            weight += self.delta_weight[i] * learning_rate
            self.weights[i] = weight

            self.biases[i] += self.delta_bias[i] * learning_rate


    def train(self, input, output, learning_rate, epochs):
        """ Function to train our model
        Args:
            input:
            output:
            learning_rate:
            epochs:            
        """
        # training the network by passing the training dataset to
        #   to the network epoch times
        delta_weight = []
        delta_bias = []
        self.input = input
        print("input head ", input.head())
        self.output = output
        print("================= Training Started ==================")

        for i in range(epochs):
            total_error = 0

            for X, y in zip(self.input, self.output):
                #input = np.expand_dims(input, 0)
                
                network_output = self.feed_forward(input)

                loss_score, loss_derivative = self.calculate_loss(output, network_output,
                                                                  self.loss_function)

                total_error += loss_score

                self.back_propogation(loss_derivative, network_output)

                self.full_gradient_descent(learning_rate)

            print(f"Error for Epoch: {i} is {total_error}")
        
        print("========== END ==========")


