import numpy as np
import matplotlib.pyplot as plt

"""
N-layer Perceptron Artificial Neural Network for Classification
"""


class NeuralNetwork:
    """
    layers - array of numbers of nodes in each layer; first number is a number of features, last number is a number of
        output nodes and everything in the middle are numbers of nodes in consecutive hidden layers
    learning_rate - float 0 - 1
    iterations - int
    """

    def __init__(self, layers, learning_rate=0.001, iterations=100):
        self.validate_parameters(layers, learning_rate, iterations)

        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.params = {}  # weights and biases
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def validate_parameters(self, layers, learning_rate, iterations):
        if len(layers) < 2 or layers is None:
            raise Exception("Layers need to have at least two numbers")

        if learning_rate <= 0 or learning_rate >= 1:
            raise Exception("Learning rate should be a float between 0 and 1")

        if iterations < 0:
            raise Exception("Iterations should be a positive integer")

    def init_weights(self):
        """
        Default random weights, based on a uniform normal distribution
        """
        np.random.seed(777)
        for index, layer in enumerate(self.layers[:-1]):
            self.params["W" + str(index + 1)] = np.random.randn(self.layers[index], self.layers[index + 1])

    def init_biases(self):
        """
        Default random biases, based on a uniform normal distribution
        """
        np.random.seed(42)
        for index, layer in enumerate(self.layers[:-1]):
            self.params["b" + str(index + 1)] = np.random.randn(self.layers[index + 1], )

    def activation_function(self, value):
        """
        ReLU
        It receives a value from a layer, which is a sum of features multiplied by corresponding weights and with added
            bias to this sum
        It returns value or 0 if a value is a negative value
        """

        return np.maximum(0, value)

    def activation_function_derivative(self, value):
        value[value <= 0] = 0
        value[value > 0] = 1
        return value

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def non_zero(self, value):
        return np.maximum(value, 0.00000001)

    '''
    Using Cross-Entropy Loss function
    https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    '''

    def cross_entropy_loss(self, actual, predicted):
        actual_size = len(actual)
        actual_inversion = 1 - actual
        predicted_inversion = 1 - predicted
        actual = self.non_zero(actual)
        predicted = self.non_zero(predicted)

        # Combined loss
        return -1 / actual_size * (
            np.sum(
                np.multiply(actual, np.log(predicted)) +
                np.multiply(actual_inversion, np.log(predicted_inversion))
            )
        )

    def forward_propagation(self):
        first_layer_result = self.X.dot(self.params["W1"]) + self.params["b1"]
        activation_function_result = self.activation_function(first_layer_result)
        second_layer_result = activation_function_result.dot(self.params["W2"]) + self.params["b2"]
        predicted = self.sigmoid(second_layer_result)
        loss = self.cross_entropy_loss(self.y, predicted)

        self.params["Z1"] = first_layer_result
        self.params["Z2"] = second_layer_result
        self.params["A1"] = activation_function_result

        return predicted, loss

    def backward_propagation(self, predicted):
        actual_inversion = 1 - self.y
        predicted_inversion = 1 - predicted

        loss_wrt_predicted = np.divide(actual_inversion, self.non_zero(predicted_inversion)) - np.divide(self.y,
                                                                                                         self.non_zero(
                                                                                                             predicted))
        loss_wrt_sigmoid = predicted * predicted_inversion
        loss_wrt_z2 = loss_wrt_predicted * loss_wrt_sigmoid

        loss_wrt_A1 = loss_wrt_z2.dot(self.params["W2"].T)
        loss_wrt_w2 = self.params["A1"].T.dot(loss_wrt_z2)
        loss_wrt_b2 = np.sum(loss_wrt_z2, axis=0, keepdims=True)

        loss_wrt_z1 = loss_wrt_A1 * self.activation_function_derivative(self.params["Z1"])
        loss_wrt_w1 = self.X.T.dot(loss_wrt_z1)
        loss_wrt_b1 = np.sum(loss_wrt_z1, axis=0, keepdims=True)

        self.update_weights(loss_wrt_w1, loss_wrt_w2)
        self.update_biases(loss_wrt_b1, loss_wrt_b2)

    def update_weights(self, loss_wrt_w1, loss_wrt_w2):
        """
        Subtract the weight derivative * learning rate
        """
        self.params["W1"] = self.params["W1"] - self.learning_rate * loss_wrt_w1
        self.params["W2"] = self.params["W2"] - self.learning_rate * loss_wrt_w2

    def update_biases(self, loss_wrt_b1, loss_wrt_b2):
        """
        Subtract the bias derivative * learning rate
        """
        self.params["b1"] = self.params["b1"] - self.learning_rate * loss_wrt_b1
        self.params["b2"] = self.params["b2"] - self.learning_rate * loss_wrt_b2

    def fit(self, X, y):
        """
        Training phase
        """

        self.X = X
        self.y = y
        self.init_weights()
        self.init_biases()

        for iteration in range(self.iterations):
            prediction, loss = self.forward_propagation()
            self.backward_propagation(prediction)
            self.loss.append(loss)

    def predict(self, X):
        Z1 = X.dot(self.params["W1"]) + self.params["b1"]
        A1 = self.activation_function(Z1)
        Z2 = A1.dot(self.params["W2"]) + self.params["b2"]
        predicted = self.sigmoid(Z2)
        return np.round(predicted)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss per iteration")
        plt.show()
