import numpy as np

class ANN:

    def __init__(self, sizes, activation_function, loss_function, gradient_type):
        """ Initializing the input variables
        Args:
            sizes: list of number of neurons in each layer
            activation_function: list of activation function from one layer to another
            loss_function: one loss function for each network
            gradient_type: one gradient type for each network
            epochs: number of epochs
        """
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.gradient_type = gradient_type

        # Defining numpy zeroes for weight and bias list for updates in back propogation
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = []
        delta_weight = []
        delta_bias = [np.random.randn(1, y) for y in sizes[1:]]

        # Taking Xaviers Weight initialization
        for i in range(self.num_of_layers-1):
            weight = np.sqrt(2/sizes[i-1]) * np.random.randn(sizes[i], sizes[i+1])
            self.weights.append(weight)
            delta_weight.append(weight)
 
        self.delta_weight = delta_weight
        self.delta_bias = delta_bias
        
        # initializing layer_outputs
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
        Args:
            z - activation i.e dot product of (layer_output, w) +b)
        Returns:
            sigmoid function of the activation
        """
        return 1.0/(1.0+np.exp(-z))


    def sigmoid_prime(self, z):
        """ Define Sigmoid Activation Function Prime
        Args:
            output - output from previous layer 
        Returns:
            activation prime - dervative of sigmoid activation function
        """
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def tanh(self, z):
        """ Define tanh Activation Function
        Args:
            z - activation i.e dot product of (layer_output, w) +b)
        Returns:
            tanh function of the activation
        """
        return np.tanh(z)


    def tanh_prime(self, z):
        """ Define tanh Activation Function prime
        Args:
            output - output from previous layer 
        Returns:
            activation prime - dervative of tanh activation function
        """
        return 1 - np.tanh(z) ** 2


    def ReLU(self, z):
        """ Define ReLU Activation Function
        Args:
            z - activation i.e dot product of (layer_output, w) +b)
        Returns:
            ReLU function of the activation
        """
        return np.maximum(0, z)

    
    def ReLU_prime(self, z):
        """ Define ReLU Activation Function prime
        Args:
            output - output from previous layer 
        Returns:
            activation prime - dervative of ReLU activation function
        """ 
        return (z > 0) * 1


    def leaky_ReLU(z):
        """ Define Leaky ReLU activation function
        Args:
            z - activation i.e dot product of (layer_output, w) +b)
        Returns:
            LeakyRelu function of the activation
        """
        if z < 0:
            return z*0.01
        else:
            return z


    def leaky_ReLU_prime(z):
        """ Define Leaky ReLU Activation Function prime
        Args:
            output - output from previous layer 
        Returns:
            activation prime - dervative of LeakyRelu activation function
        """
        if z < 0:
            return 0.01
        else:
            return 1

    ######################################################
    #    Define all cost functions and its prime   #     
    ######################################################
    def MSE(self, predicted, actual, derivative =False):
        """ Define MSE Cost Function
        Args:
            output: y-hat (predicted value)
            input: y (actual value)
        Returns:
            loss_score
            loss_derivative
        """
        if derivative:
            loss_compute = np.mean(predicted-actual, axis=0)
            loss_derivative = 2*(loss_compute)
            return loss_derivative
        else:
            loss_compute = np.mean(predicted-actual, axis=0)
            loss_score = np.square(loss_compute)
            return loss_score


    def binary_cross_entropy(self, predicted, actual, derivative=False):
        """ Define Binary Cross entropy function
        Args:
            predicted: y-hat (predicted value)
            actual: y (actual value)
         Returns:
            loss_score - Error of the whole network
            loss_derivative - Derivative of the score
        """
        # print("predicted output", predicted)
        # print("actual output", actual)
        if derivative:
            loss_derivative = -(actual/predicted - (1-actual)/(1-predicted))
            return loss_derivative
        else:
            loss_score = -np.mean((actual * np.log(predicted)) + ((1-actual)*np.log(1-predicted)))
            return loss_score
    

    # requires changes as this might not be the correct formula - may be find another one - task on Rucha
    def cross_entropy(self, predicted, actual, derivative=False):
        """ Define cross entropy function
        Args:
            predicted: y-hat (predicted value)
            actual: y (actual value)
         Returns:
            loss_score - Error of the whole network
            loss_derivative - Derivative of the score
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
            actual: Actual output given in the dataset
            predicted: Predicted output using feedforward
            loss_function: Loss function to calculate error of the network
        Returns:
           loss_score - Error of the whole network
            loss_derivative - Derivative of the score
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
        Args:
            Input - Input from one layer to another
        Returns:
            Layer_Output - Output from Layer to another
        """
        # pass our inputs through our neural network
        activation_index = 0
        layer_output = input
        self.layer_outputs[0] = layer_output

        for b, w in zip(self.biases, self.weights):
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
            activation_index+=1
            
        return layer_output


    def back_propogation(self, loss_derivative, output):
        """ Define Back Propogation
        Args:
            loss_derivative: 
            output: 
        Returns:

        """
        for i in reversed(range(self.num_of_layers-1)):
            output = self.layer_outputs[i+1]
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
            current_layer_output = np.asarray(self.layer_outputs[i])
            output_reshaped = current_layer_output.reshape(current_layer_output.shape[0], -1)

            #currently issues with this hence commented
            #self.delta_weight[i] = np.dot(output_reshaped, delta_reshaped)

            #Getting the delta bias as delta
            self.delta_bias[i] = delta

            #updating the loss derivative to pass to next layers 
            loss_derivative = np.dot(loss_derivative, self.weights[i].T)
    

    def full_gradient_descent(self, learning_rate):
        """ Minimise the gradient to find optima 
        Args: 
            delta_weight:
            delta_bias:
            learning_rate:
        """
        for i in range(self.num_of_layers-1):
            weight = self.weights[i]
            weight += self.delta_weight[i] * learning_rate
            self.weights[i] = weight

            self.biases[i] += self.biases[i] * learning_rate

    def train(self, input, output, learning_rate, epochs):
        """ Function to train the model
        Args:
            input:
            output:
            learning_rate:
            epochs:  
        Returns:
            Network_output - Predicted values to be used to check the accuracy         
        """
        # training the network by passing the training dataset to
        # the network epoch times
        print("Training the network...")
        self.input = input
        self.output = output
        print("input shape", input.shape)
        print("output shape", output.shape)

        for i in range(epochs):
            total_error = 0

            for X, y in zip(input, output):
               
                
                network_output = self.feed_forward(input)

                loss_score, loss_derivative = self.calculate_loss(y, network_output,
                                                                    self.loss_function)

                #calculating the total error from all the training epochs
                total_error += loss_score

                #Back propogate to update weights biases and errors
                self.back_propogation(loss_derivative, network_output)
                
                #Gradient Descent to minimize the gradient 
                self.full_gradient_descent(learning_rate)

            if i%10 == 0 or i==99:
                print(f"Error for Epoch: {i} is {total_error}")
        
        print("========== END ==========")
        return network_output

