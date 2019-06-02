# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def trainNetwork(w,train, test):
    epoch = 0
    last_w = pd.DataFrame([0])

    # while abs(w.iloc[:,:].sum().sum() - last_w.iloc[0].values[0]) > 0.0005: 
    while epoch < 100:
        last_w.iloc[0] = w.iloc[:,:].sum().sum()

        for i in range(len(train)):

            distances = pd.DataFrame(np.zeros(number_neurons_1st_layer))
            for neuron in range(number_neurons_1st_layer):
                for column in range(len(train.columns) - 1):
                    distances.iloc[neuron] += (train.iloc[i, column] - w.iloc[neuron,column]) * (train.iloc[i, column] - w.iloc[neuron,column])
                distances.iloc[neuron] = math.sqrt(distances.iloc[neuron])

            for column in range(len(train.columns) - 1):
                w.iloc[distances[:].idxmin().values[0], column] += learning_rate * (train.iloc[i,column] - w.iloc[distances[:].idxmin().values[0], column])

            for neuron in neighborhood[distances[:].idxmin().values[0]]:
                normalization_sum = 0
                for column in range(len(train.columns) - 1):
                    w.iloc[neuron, column] += (learning_rate/2) * (train.iloc[i,column] - w.iloc[neuron, column])

            
            for neuron in range(number_neurons_1st_layer):
                normalization_sum = 0
                for column in range(len(train.columns) - 1):
                    normalization_sum += w.iloc[neuron, column] * w.iloc[neuron, column]
                normalization_sum = math.sqrt(normalization_sum)
                for column in range(len(train.columns) - 1):
                    w.iloc[neuron, column] = w.iloc[neuron, column]/normalization_sum

        print "[EPOCH] " + str(epoch) + " [STATUS] " + str(abs(w.iloc[:,:].sum().sum() - last_w.iloc[0].values[0]))
        epoch += 1

    for i in range(len(train)):
        distances = pd.DataFrame(np.zeros(number_neurons_1st_layer))
        for neuron in range(number_neurons_1st_layer):
            for column in range(len(train.columns) - 1):
                distances.iloc[neuron] += (train.iloc[i, column] - w.iloc[neuron,column]) * (train.iloc[i, column] - w.iloc[neuron,column])
            distances.iloc[neuron] = math.sqrt(distances.iloc[neuron])
        print  str(i+1) + " & " + str(distances[:].idxmin().values[0]) + "\\\\"

    print "AMOSTRAS"

    for i in range(len(test)):
        distances = pd.DataFrame(np.zeros(number_neurons_1st_layer))
        for neuron in range(number_neurons_1st_layer):
            for column in range(len(test.columns)):
                distances.iloc[neuron] += (test.iloc[i, column] - w.iloc[neuron,column]) * (test.iloc[i, column] - w.iloc[neuron,column])
            distances.iloc[neuron] = math.sqrt(distances.iloc[neuron])
        print str(i+1) + " & " + str(test.iloc[i,0]) + " & " + str(test.iloc[i,1]) + " & " + str(test.iloc[i,2]) + " & " + str(distances[:].idxmin().values[0]) + " \\\\"

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

learning_rate = 0.001
number_neurons_1st_layer = 16

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

train = pd.read_csv('./train.csv', header = None)
test = pd.read_csv('./test.csv', header = None)

neighborhood = [[1,4],[0,2,5],[1,3,6],[2,7],[0,5,8],[1,4,6,9],[2,5,7,10],[3,6,11],[4,11,12],[5,8,10,13],[6,9,11,14],[7,10,15],[8,13],[9,12,14],[10,13,15],[11,14]]

w = pd.DataFrame(train.iloc[:number_neurons_1st_layer,:-1])
# w = pd.DataFrame(np.random.rand(number_neurons_1st_layer, 3))

trainNetwork(w, train, test)
