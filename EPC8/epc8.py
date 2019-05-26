# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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

def kMeans(train, w1):

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

    print str(groups) + "\n" + str(variance)

    return w1, variance


#////////////////////////////////////////////////////////////////////
# OUTPUT LAYER //////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def outputLayer(train,w1,w2, variance):
    d = train.iloc[:,2]
    x = train.iloc[:,:-1]

    z = pd.DataFrame(np.zeros((len(train),3)))
    for i in range(len(z)):
        z.iloc[i,2] -= 1

    for i in range(len(train)):
        for j in range(number_neurons_1st_layer):
            z.iloc[i,j] = math.exp(-(math.pow(x.iloc[i,0] - w1.iloc[j,0], 2) + math.pow(x.iloc[i,1] - w1.iloc[j,1], 2))/(2*variance[j]))

    epoch = 0
    Eqm = 0
    lEqm = 1 
    while(abs(Eqm - lEqm) > error_tolerance and epoch < 1000):
        lEqm = Eqm

        # SECOND LAYER WEIGHTS
        for i in range(len(train)):
            y = (z.iloc[i,:] * w2.iloc[:,0]).sum()

            for j in range(number_neurons_1st_layer + 1):
                gradient = (d.iloc[i] - y) * dlogistic(z.iloc[i,j])
                w2.iloc[j,0] += learning_rate * gradient * z.iloc[i,j]
        
        # QUADRATIC ERROR
        Eqm = 0
        for i in range(len(train)):
            y = (z.iloc[i,:] * w2.iloc[:,0]).sum()
            Eqm += math.pow(d.iloc[i] - y ,2)/2
        Eqm = Eqm/len(train) 
        print epoch
        print "[QUADRATIC ERROR] " + str(abs(Eqm - lEqm))

        epoch += 1
    
    return w2

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

def runTest(train,w1,w2, variance):
    x = train.iloc[:,:-1]

    z = pd.DataFrame(np.zeros((len(train),3)))
    for i in range(len(z)):
        z.iloc[i,2] -= 1

    for i in range(len(train)):
        for j in range(number_neurons_1st_layer):
            z.iloc[i,j] = math.exp(-(math.pow(x.iloc[i,0] - w1.iloc[j,0], 2) + math.pow(x.iloc[i,1] - w1.iloc[j,1], 2))/(2*variance[j])) 

    for i in range(len(train)):
        y = (z.iloc[i,:] * w2.iloc[:,0]).sum()
        if y > 0:
            print "1 " + str(y)
        else:
            print "-1 " + str(y)

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# TRAIN DATASET ----------------------------------------------------
train = pd.read_csv('./train.csv', header = None)
test = pd.read_csv('./test.csv', header = None)

w1 = pd.DataFrame(train.iloc[:number_neurons_1st_layer,:-1])
w2 = pd.DataFrame(np.random.rand(number_neurons_1st_layer + 1, number_neurons_2nd_layer))

# CALCULATE CLUSTERS
w1, variance = kMeans(train, w1)
w2, outputLayer(train,w1,w2,variance)
runTest(test, w1, w2, variance)
print "[COORDINATES W1] " + str(w1)
print "[COORDINATES W2] " + str(w2)
print "[VARIANCE] " + str(variance)