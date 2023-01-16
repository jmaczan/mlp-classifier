from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(13, 8, 1), max_iter=1000, tol=0.001, )