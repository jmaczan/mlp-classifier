import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nn

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

print(f"Train set: {Xtrain.shape}")
print(f"Train labels: {ytrain.shape}")
print(f"Test set: {Xtest.shape}")
print(f"Test labels: {ytest.shape}")

neural_network = nn.NeuralNetwork([13, 1], iterations=10)
neural_network.fit(Xtrain, ytrain)
neural_network.plot_loss()
print(f"Loss: {neural_network.loss}")

neural_network = nn.NeuralNetwork([13, 1], iterations=100)
neural_network.fit(Xtrain, ytrain)
neural_network.plot_loss()
print(f"Loss: {neural_network.loss}")

neural_network = nn.NeuralNetwork([13, 8, 1])
neural_network.fit(Xtrain, ytrain)
neural_network.plot_loss()
print(f"Loss: {neural_network.loss}")

neural_network = nn.NeuralNetwork([13, 8, 1], iterations=1000)
neural_network.fit(Xtrain, ytrain)
neural_network.plot_loss()
print(f"Loss: {neural_network.loss}")
