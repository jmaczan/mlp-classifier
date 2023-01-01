# INF-Z-2023-Jedrzej-Maczan-285809

[Neural network implementation](nn.py)

## Report

### Neuron

It accepts list of variables as an input. Those variables are being multiplied by corresponding weights. The result of
multiplication is summed and provided as an input to the activation function, which calculates an output number

### Perceptron

It's a neural network build with a single neuron

### Neural network

Consists of multiple neurons

### Layer

Layer is a group of neurons

Layers between input and output layers are called hidden layers

### First layer - input layer

It is not counted as a layer in a network

Number of nodes in the first layer equals number of features in the input data

### Last layer - output layer

Number of nodes (variables) in output layer depends on type of desired prediction

### Weight

It represents how a given feature is important. It is multiplied by a feature's value

### Bias

It is a starting value for a given neuron. It is added to sum of multiplications of weights and features values.

### Activation function

It determines whether a neuron's contribution to the neural network should be taken into account or not

### Forward propagation

It is a training process for a network. It consists of the following steps:
1. Multiply each input feature and randomly generated corresponding weight of the first layer, sum them up and add the bias
2. Use this result as an input to activation function 
3. Use the output from activation function as a features for the next weights and repeat this step until the last layer
4. Pass the last result to the output activation function
5. Compute the loss function by using last result (prediction) and actual (true) values 

### Backward propagation

It is about learning process in neural network through improving network's weights and biases.

A network checks the output for various weights and evaluates them using loss function. Decrease of loss means that weights are getting better

Backpropagation uses derivatives of loss with respect to all previously calculated values - weights, biases and activation function results

Input values are not differentiated

Backward propagation steps:
1. Derivative with respect to activation function $\frac{\partial L}{\partial a}$

### Optimization

It means looking for the best possible weights and biases in the network