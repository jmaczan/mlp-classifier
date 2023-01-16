import numpy as np
import matplotlib.pyplot as plt

"""
N-layer Perceptron Artificial Neural Network for Classification tasks
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
        self.computed_layer_sums = np.array([None] * (len(layers) - 1))
        self.computed_activation_functions = np.array([None] * (len(layers) - 1))
        self.biases = [0.0] * (len(layers) - 1)
        self.weights = [0.0] * (len(layers) - 1)
        self.loss = []
        self.train_set = None
        self.train_labels = None

    @staticmethod
    def validate_parameters(layers, learning_rate, iterations):
        if len(layers) < 2 or layers is None:
            raise Exception("Layers need to have at least two numbers")

        if learning_rate <= 0 or learning_rate >= 1:
            raise Exception("Learning rate should be a float between 0 and 1")

        if iterations < 0:
            raise Exception("Iterations should be a positive integer")

    def init_weights(self):
        np.random.seed(777)
        for index, layer in enumerate(self.layers[:-1]):
            self.weights[index] = np.random.randn(self.layers[index], self.layers[index + 1])

    def init_biases(self):
        np.random.seed(42)
        for index, layer in enumerate(self.layers[:-1]):
            self.biases[index] = np.random.randn(self.layers[index + 1], )

    @staticmethod
    def activation_function(value):
        return np.maximum(0, value)

    @staticmethod
    def activation_function_derivative(value):
        value[value <= 0] = 0
        value[value > 0] = 1
        return value

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def non_zero(value):
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
        for index, layer in enumerate(self.layers[:-1]):
            if index == 0:
                self.computed_layer_sums[index] = self.train_set.dot(self.weights[index]) + self.biases[index]
                continue

            activation_function_result = self.activation_function(self.computed_layer_sums[index - 1])
            self.computed_activation_functions[index - 1] = activation_function_result

            self.computed_layer_sums[index] = activation_function_result.dot(self.weights[index]) + self.biases[index]

        predicted = self.sigmoid(self.computed_layer_sums[(len(self.computed_layer_sums) - 1)])
        loss = self.cross_entropy_loss(self.train_labels, predicted)

        return predicted, loss

    def backward_propagation(self, predicted):
        actual_inversion = 1 - self.train_labels
        predicted_inversion = 1 - predicted

        # in reversed order - from last to first
        loss_layer_sums = np.array([None] * (len(self.layers) - 1))
        loss_activation_functions = np.array([None] * (len(self.layers) - 1))
        loss_layer_weights = np.array([None] * (len(self.layers) - 1))
        loss_layer_biases = np.array([None] * (len(self.layers) - 1))

        layers = self.layers[:-1]
        layers.reverse()
        for index, layer in enumerate(layers):
            if index == 0:  # last layer
                loss_predicted = np.divide(actual_inversion, self.non_zero(predicted_inversion)) - np.divide(
                    self.train_labels,
                    self.non_zero(
                        predicted))
                loss_sigmoid = predicted * predicted_inversion
                loss_layer_sums[0] = loss_predicted * loss_sigmoid
                continue

            loss_activation_functions[index - 1] = loss_layer_sums[index - 1].dot(self.weights[index].T)
            loss_layer_weights[index - 1] = self.computed_activation_functions[index - 1].T.dot(
                loss_layer_sums[index - 1])
            loss_layer_biases[index - 1] = np.sum(loss_layer_sums[index - 1], axis=0, keepdims=True)
            loss_layer_sums[index] = loss_activation_functions[index - 1] * self.activation_function_derivative(
                self.computed_layer_sums[index - 1])

        loss_layer_weights[(len(loss_layer_weights) - 1)] = self.train_set.T.dot(
            loss_layer_sums[(len(loss_layer_sums) - 1)])
        loss_layer_biases[(len(loss_layer_biases) - 1)] = np.sum(loss_layer_sums[(len(loss_layer_sums) - 1)], axis=0,
                                                                 keepdims=True)

        self.update_weights(loss_layer_weights)
        self.update_biases(loss_layer_biases)

    def update_weights(self, loss_layer_weights):
        """
        Subtract the weight with derivative * learning rate
        """

        weights = loss_layer_weights[::-1]

        for index, weight in enumerate(weights):
            self.weights[index] = self.weights[index] - self.learning_rate * weight

    def update_biases(self, loss_layer_biases):
        """
        Subtract the bias with derivative * learning rate
        """

        biases = loss_layer_biases[::-1]

        for index, bias in enumerate(biases):
            self.biases[index] = self.biases[index] - self.learning_rate * bias

    def fit(self, train_set, train_labels):
        self.train_set = train_set
        self.train_labels = train_labels
        self.init_weights()
        self.init_biases()

        for iteration in range(self.iterations):
            prediction, loss = self.forward_propagation()
            self.backward_propagation(prediction)
            self.loss.append(loss)

    def predict(self, data):
        for index, layer in enumerate(self.layers[:-1]):
            if index == 0:
                self.computed_layer_sums[index] = data.dot(self.weights[index]) + self.biases[index]
                continue

            activation_function_result = self.activation_function(self.computed_layer_sums[index - 1])
            self.computed_activation_functions[index - 1] = activation_function_result

            self.computed_layer_sums[index] = activation_function_result.dot(self.weights[index]) + self.biases[index]

        return np.round(self.sigmoid(self.computed_layer_sums[(len(self.computed_layer_sums) - 1)]))

    @staticmethod
    def accuracy(actual, predicted):
        return int(sum(actual == predicted) / len(actual) * 100)

    def plot(self):
        plt.plot(self.loss)
        plt.title("Loss per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
