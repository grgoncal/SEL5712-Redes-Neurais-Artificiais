# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

error_tolerance = 0.0000001
learning_rate = 0.02
number_neurons_1st_layer = 2
number_neurons_2nd_layer = 1

#////////////////////////////////////////////////////////////////////
# K-MEANS ///////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def kMeans(train, w1):

    last_groups = [[2],[0]]
    groups = [[],[]]

    while(last_groups != groups):
        
        last_groups = groups
        euclidian_distance = pd.DataFrame(np.zeros((len(train), number_neurons_1st_layer)))
        groups = [[],[]]

        for i in range(len(train)):
            for j in range(number_neurons_1st_layer):
                euclidian_distance.iloc[i,j] = math.sqrt(math.pow(train.iloc[i,0] - w1.iloc[j,0], 2) + math.pow(train.iloc[i,1] - w1.iloc[j,1], 2))
            groups[euclidian_distance.iloc[i,:].idxmin()].append(i)

        w1 = pd.DataFrame(np.zeros((2,2)))
        
        for i in groups[0]:
            for j in range(number_neurons_1st_layer):
                w1.iloc[0,j] += (train.iloc[i,j])/len(groups[0])

        for i in groups[1]:
            for j in range(number_neurons_1st_layer):
                w1.iloc[1,j] += (train.iloc[i,j])/len(groups[1])
    
    variance = [0,0]

    for i in groups[0]:
        variance[0] +=  math.pow(train.iloc[i,0] - w1.iloc[0,0], 2) + math.pow(train.iloc[i,1] - w1.iloc[0,1], 2)
    variance[0] = variance[0]/len(groups[0])

    for i in groups[1]:
        variance[1] +=  math.pow(train.iloc[i,0] - w1.iloc[1,0], 2) + math.pow(train.iloc[i,1] - w1.iloc[1,1], 2)
    variance[1] = variance[1]/len(groups[1])

    return w1, variance


#////////////////////////////////////////////////////////////////////
# OUTPUT LAYER //////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def outputLayer(train,w1,w2, variance):
    d = train.iloc[:,2]
    x = train.iloc[:,:-1]

    z = pd.DataFrame(np.zeros((len(train),2)))

    for i in range(len(train)):
        for j in range(number_neurons_1st_layer):
            z.iloc[i,j] = -(math.pow(x.iloc[i,0] - w1.iloc[j,0], 2) + math.pow(x.iloc[i,1] - w1.iloc[j,1], 2))/(2*math.pow(variance[j], 2)) 

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# TRAIN DATASET ----------------------------------------------------
train = pd.read_csv('./train.csv', header = None)

w1 = pd.DataFrame(train.iloc[:number_neurons_1st_layer,:-1])
w2 = pd.DataFrame(np.random.rand(number_neurons_1st_layer, number_neurons_2nd_layer))

# CALCULATE CLUSTERS
w1, variance = kMeans(train, w1)
outputLayer(train,w1,w2,variance)