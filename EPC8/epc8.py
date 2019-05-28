# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

error_tolerance = 0.0000001
learning_rate = 0.01
number_neurons_1st_layer = 5
number_neurons_2nd_layer = 1

#////////////////////////////////////////////////////////////////////
# K-MEANS ///////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def kMeans(train, w1, number_neurons_1st_layer):

    groups = []
    last_groups = []
    for i in range(number_neurons_1st_layer):
        last_groups.append([1])
        groups.append([])
    

    while(last_groups != groups):

        last_groups = groups
        groups = []
        for i in range(number_neurons_1st_layer):
            groups.append([])
        
        euclidian_distance = pd.DataFrame(np.zeros((len(train), number_neurons_1st_layer)))

        for i in range(len(train)):
            for j in range(number_neurons_1st_layer):
                for column in range(len(train.columns) - 1):
                    euclidian_distance.iloc[i,j] += math.pow(train.iloc[i,column] - w1.iloc[j,column] , 2)
                euclidian_distance.iloc[i,j] = math.sqrt(euclidian_distance.iloc[i,j])
            groups[euclidian_distance.iloc[i,:].idxmin()].append(i)

        w1 = pd.DataFrame(np.zeros((number_neurons_1st_layer,3)))
        
        for i in range(number_neurons_1st_layer):
            for j in groups[i]:
                for n in range(len(train.columns) - 1):
                    w1.iloc[i,n] += train.iloc[j,n]/len(groups[i])

    
    variance = []
    for i in range(number_neurons_1st_layer):
        variance.append(0)

    for i in range(number_neurons_1st_layer):
        for j in groups[i]:
            for column in range(len(train.columns) - 1):
                variance[i] += math.pow(train.iloc[j,column] - w1.iloc[i,column], 2) 
        variance[i] = variance[i]/len(groups[i])

    return w1, variance


#////////////////////////////////////////////////////////////////////
# OUTPUT LAYER //////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def outputLayer(train, w1, w2, variance, testNumber, number_neurons_1st_layer):
    chart = [[],[]]

    d = train.iloc[:,3]
    x = train.iloc[:,:-1]

    z = pd.DataFrame(np.zeros((len(train), number_neurons_1st_layer + 1)))
    for i in range(len(z)):
        z.iloc[i,number_neurons_1st_layer] -= 1

    for sample in range(len(train)):
        for neuron in range(number_neurons_1st_layer):
            for column in range(len(x.columns)):
                z.iloc[sample,neuron] += math.pow(x.iloc[sample,column] - w1.iloc[neuron,column], 2)
            z.iloc[sample,neuron] = math.exp(-z.iloc[sample,neuron]/(2*variance[neuron]))

    epoch = 0
    Eqm = 0
    lEqm = 1 
    while(abs(Eqm - lEqm) > error_tolerance and epoch < 5000):
        lEqm = Eqm

        # SECOND LAYER WEIGHTS
        for i in range(len(train)):
            y = logistic((z.iloc[i,:] * w2.iloc[:,0]).sum())

            for j in range(number_neurons_1st_layer + 1):
                gradient = (d.iloc[i] - y) * dlogistic((z.iloc[i,:]).sum())
                w2.iloc[j,0] += learning_rate * gradient * z.iloc[i,j]
        
        # QUADRATIC ERROR
        Eqm = 0
        for i in range(len(train)):
            y = (z.iloc[i,:] * w2.iloc[:,0]).sum()
            Eqm += math.pow(d.iloc[i] - y ,2)/2
        Eqm = Eqm/len(train) 

        chart[0].append(epoch)
        chart[1].append(Eqm)

        epoch += 1
        print str(Eqm) + " " + str(abs(Eqm - lEqm))

    plt.figure(1)
    plt.plot(chart[0][1:], chart[1][1:])
    plt.xlabel('Epochs')
    plt.ylabel('EQM')
    plt.savefig("./" + str(number_neurons_1st_layer) + "_test_number_" + str(testNumber) + ".png", dpi = 500)
    plt.close()

    return w2, epoch, Eqm

#////////////////////////////////////////////////////////////////////
# LOGISTIC FUNCTION /////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def logistic(x):
    return 1 / (1 + math.exp(-x))

#////////////////////////////////////////////////////////////////////
# DERIVATIVE LOGISTIC FUNCTION //////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def dlogistic(x):
    return logistic(x) * (1 - logistic(x))

#////////////////////////////////////////////////////////////////////
# TEST SAMPLES //////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def runTest(test,w1,w2, variance, number_neurons_1st_layer):
    x = test.iloc[:,:]

    z = pd.DataFrame(np.zeros((len(test), number_neurons_1st_layer + 1)))
    for i in range(len(z)):
        z.iloc[i,number_neurons_1st_layer] -= 1

    for sample in range(len(test)):
        for neuron in range(number_neurons_1st_layer):
            for column in range(len(x.columns) - 1):
                z.iloc[sample,neuron] += math.pow(x.iloc[sample,column] - w1.iloc[neuron,column], 2)
            z.iloc[sample,neuron] = math.exp(-z.iloc[sample,neuron]/(2*variance[neuron]))

            
    out = []

    for i in range(len(test)):
        y = logistic((z.iloc[i,:] * w2.iloc[:,0]).sum())
        out.append(y)

    return out

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# TRAIN DATASET ----------------------------------------------------
train = pd.read_csv('./train.csv', header = None)
test = pd.read_csv('./test.csv', header = None)

results = []

for number_neurons in range(5,20,5):
    number_neurons_1st_layer = number_neurons
    for test_number in range(1,4):
        w1 = pd.DataFrame(train.iloc[:number_neurons_1st_layer,:-1])
        w2 = pd.DataFrame(np.random.rand(number_neurons_1st_layer + 1, number_neurons_2nd_layer))

        # CALCULATE CLUSTERS
        w1, variance = kMeans(train, w1,number_neurons_1st_layer)
        w2, epoch, eqm = outputLayer(train,w1,w2,variance,test_number, number_neurons_1st_layer)
        
        index_results = (number_neurons/5) * 3 - (4 - test_number)
        results.append(runTest(test, w1, w2, variance, number_neurons_1st_layer))
        print "END TRAINING " + str(test_number) + " NUMBER OF EPOCHS: " + str(epoch) + " EQM: " + str(eqm)
        print str(results[index_results])

print "[RESUTLADOS] ////////////////////////////////////////////////////////////"

print str(results)

