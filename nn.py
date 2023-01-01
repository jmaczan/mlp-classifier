import numpy as np


class NeuralNetwork:
    """
    layers - array of numbers of nodes in each layer; first number is a number of features, last number is a number of
        output nodes and everything in the middle are numbers of nodes in consecutive hidden layers
    learning_rate - float 0 - 1
    iterations - int
    """

    def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=100):
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.params = {}  # weights and biases
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        """
        Default random weights, based on a uniform normal distribution
        """
        np.random.seed(777)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])  # Array 13 x 8 for default layers
        self.params["W2"] = np.random.randn(self.layers[1], self.layers[2])  # Array 8 x 1 (?) for default layers

    def init_biases(self):
        """
        Default random biases, based on a uniform normal distribution
        """
        np.random.seed(42)
        self.params["b1"] = np.random.randn(self.layers[1], )  # Vector of 8 elements for default layers
        self.params["b2"] = np.random.randn(self.layers[2], )  # Single element vector, for defaults

    def activation_function(self, value):
        """
        ReLU
        It receives a value from a layer, which is a sum of features multiplied by corresponding weights and with added
            bias to this sum
        It returns value or 0 if a value is a negative value
        """

        return np.maximum(0, value)

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

        loss_wrt_predicted = np.divide(actual_inversion, self.non_zero(predicted_inversion)) - np.divide(self.y, self.non_zero(predicted))
        loss_wrt_sigmoid = predicted * predicted_inversion
        loss_wrt_z2 = loss_wrt_predicted * loss_wrt_sigmoid

        loss_wrt_A1 = loss_wrt_z2.dot(self.params["W2"].T)
        loss_wrt_w2 = self.params["A1"].T.dot(loss_wrt_z2)
        loss_wrt_b2 = np.sum(loss_wrt_z2, axis=0, keepdims=True)

        loss_wrt_z1 = loss_wrt_A1 * self.activation_function(self.params["Z1"])
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

