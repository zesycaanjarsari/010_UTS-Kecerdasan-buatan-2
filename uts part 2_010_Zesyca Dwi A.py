# Zesyca Dwi Anjarsari
# 21091397010
# Multiple perceptron / Neuron batch and multiple layer 2

# menginisialisasi numpy
import numpy as np

# menginisialisasi variabel
# menginput nilai variabel layer feature 10 dan batch 6
inputs = [
    [4.7, 4.5, 5.0, 3.0, 2.1, 4.3, 4.7, 4.5, 2.0, 2.5],
    [2.0, 3.0, 2.1, 4.3, 4.5, 2.3, 3.5, 7.5, 3.4, 6.5],
    [3.2, 1.2, 4.3, 0.6, 1.5, 6.2, 2.4, 3.2, 1.5, 3.4],
    [4.5, 2.3, 3.5, 0.8, 4.6, 1.2, 9.2, 9.4, 4.7, 4.5],
    [3.6, 3.8, 4.5, 3.6, 2.0, 3.5, 4.7, 4.5, 5.0, 5.5],
    [1.5, 2.3, 8.5, 9.3, 4.2, 1.2, 4.3, 0.6, 1.5, 6.2],
]
# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# menginput jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5
weights1 = [
    [1.0, 1.5, 2.0, 2.5, 3.7, 3.5, 4.7, 4.5, 5.0, 5.5],
    [1.5, 1.4, 2.2, 2.4, 3.2, 3.4, 4.2, 4.4, 5.2, 5.4],
    [2.7, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [2.5, 6.4, 7.2, 7.4, 8.2, 8.4, 9.2, 9.4, 10.2, 10.4],
    [3.5, 18.5, 18.0, 20.5, 30.0, 30.5, 40.0, 40.5, 50.0, 50.5],
]
# menginisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu layer 1 = 5 neuron
biases1 = [1.5, 2.3, 3.1, 4.7, 5.8]

# menginisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# menginput jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [
    [6.3, 6.4, 2.2, 1.2, 5.2],
    [2.0, 5.0, 3.2, 6.4, 2.3],
    [7.1, 3.2, 8.5, 4.3, 2.4]]

# menginisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3
biases2 = [2.4, 4.3, 5.9]
transpose = np.dot(inputs, np.array(weights1).T)
print(transpose)
# output
# membuat perhitungan layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# membuat perhitungan layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print output layer2
print(layer2_outputs)
