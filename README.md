# Multi-Layer Perceptron Classifier 🤗

[Implementation of Multi-Layer Perceptron Artificial Neural Network in Python 3](nn.py)

## About

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

Backward propagation steps, based on Patrick David's [All the Backpropagation derivatives](https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60):
1. Derivative with respect to (wrt) activation function $\frac{\partial L}{\partial a}$: 

Derivative of the negative log likelihood function (cross-entropy):

$$[yln(a) + (1-y)ln(1-a)] = $$

$$[-yln(a) - (1-y)ln(1-a)] = $$

$$\frac{\partial L}{\partial a} = [\frac{-y}{a} - (-)\frac{(1-y)}{(1-a)}]$$

$$\frac{\partial L}{\partial a} = [\frac{-y}{a} + \frac{(1-y)}{(1-a)}]$$

2. Derivative of sigmoid $\frac{\partial a}{\partial z}$:

Derivative is 
$$\frac{1}{1+e^{-z}} = ... = sig(z) * (1 - sig (z))$$

3. Derivative wrt linear function

Linear function is $z = W*X + b$, where $W$ represents weights, $X$ is input and $b$ is bias

Derivative is $a - y$

4. Derivative wrt weights $\frac{\partial z}{\partial w}$:

$z = w^T*X + b$ 

Derivative is $x$

5. Derivative wrt bias $\frac{\partial L}{\partial b}$:

Derivative is $a - y$

### Optimization

It means looking for the best possible weights and biases in the network

### Training

Repeat these three steps:
1. Forward propagation
2. Backward propagation
3. Update weights with calculated gradients


## Resources
When implementing the network, I based on these great online resources:
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
- https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432
- https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-2-testing-the-network-c1f0c1c9cbb0
- https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
- https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

## Author

© Copyright [Jędrzej Paweł Maczan](https://maczan.pl/). Made in [Poland](https://en.wikipedia.org/wiki/Poland), 2022 - 2023
