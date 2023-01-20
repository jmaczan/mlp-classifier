import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nn import NeuralNetwork

headers = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
           'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
           'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
           'num_of_major_vessels', 'thal', 'heart_disease']

data = pd.read_csv('heart.dat', sep=' ', names=headers)

test_data = data.drop(columns=['heart_disease'])

data['heart_disease'] = data['heart_disease'].replace(1, 0)
data['heart_disease'] = data['heart_disease'].replace(2, 1)

y_label = data['heart_disease'].values.reshape(test_data.shape[0], 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(test_data, y_label, test_size=0.2, random_state=2)

sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)


def lowest_loss(loss):
    return [x for x in loss if x == x][-1]


print(f"Train set: {Xtrain.shape}")
print(f"Train labels: {ytrain.shape}")
print(f"Test set: {Xtest.shape}")
print(f"Test labels: {ytest.shape}")

mlp_clf = MLPClassifier(hidden_layer_sizes=(13, 8), max_iter=100, tol=0.1)
mlp_clf.fit(Xtrain, ytrain)
prediction = mlp_clf.predict(Xtest)
train_pred = mlp_clf.predict(Xtrain)
print('Test accuracy', metrics.accuracy_score(prediction, ytest))
print('Train accuracy:', metrics.accuracy_score(train_pred, ytrain))


class Benchmark:
    def __init__(self, layers, learning_rate=0.001, iterations=100):
        neural_network = NeuralNetwork(layers, learning_rate, iterations)
        neural_network.fit(Xtrain, ytrain)
        neural_network.plot()
        train_prediction = neural_network.predict(Xtrain)
        test_prediction = neural_network.predict(Xtest)
        print(f"""Layers: {layers},
Iterations: {iterations},
Learning rate: {learning_rate},
Lowest loss: {lowest_loss(neural_network.loss)},
Accuracy for train set: {neural_network.accuracy(ytrain, train_prediction)},
Accuracy for test set: {neural_network.accuracy(ytest, test_prediction)}""")


# how iterations affect results
Benchmark(layers=[13, 8, 1], iterations=10)
Benchmark(layers=[13, 8, 1], iterations=100)
Benchmark(layers=[13, 8, 1], iterations=500)
Benchmark(layers=[13, 8, 1], iterations=1000)

# how layers affect results
Benchmark(layers=[13, 1], iterations=100)
Benchmark(layers=[13, 3, 1], iterations=100)
Benchmark(layers=[13, 8, 1], iterations=100)
Benchmark(layers=[13, 13, 1], iterations=100)
Benchmark(layers=[13, 30, 1], iterations=100)
Benchmark(layers=[13, 50, 1], iterations=100)
Benchmark(layers=[13, 100, 1], iterations=100)

# how learning rate affects results
Benchmark(layers=[13, 8, 1], iterations=100, learning_rate=0.9999)
Benchmark(layers=[13, 8, 1], iterations=100, learning_rate=0.1)
Benchmark(layers=[13, 8, 1], iterations=100, learning_rate=0.01)
Benchmark(layers=[13, 8, 1], iterations=100, learning_rate=0.001)
Benchmark(layers=[13, 8, 1], iterations=100, learning_rate=0.0001)

Benchmark(layers=[13, 1], iterations=1000, learning_rate=0.01)
Benchmark(layers=[13, 1], iterations=10000, learning_rate=0.01)
