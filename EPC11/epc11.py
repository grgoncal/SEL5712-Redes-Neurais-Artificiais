#////////////////////////////////////////////////////////////////////
# IMPORTS ///////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def trainLVQ1(w,train):
    global NUMBER_OF_CLASSES
    global LEARNING_RATE

    EPOCH = 0
    last_w = 0

    while(EPOCH < 5000 and (w.values.sum() != last_w)) :
        # SAVE LAST WEIGHTS
        last_w = w.values.sum()

        # EUCLIDIAN DISTANCES INITIALIZATION [NO OF TRAIN DATA, NO OF NEURONS]
        distances = pd.DataFrame(np.zeros([train.shape[0], NUMBER_OF_CLASSES]))

        # CALCULATE DISTANCES
        for sample in range(train.shape[0]): # 0 to 15
            for neuron in range(NUMBER_OF_CLASSES): # 0 to 3
                for column in range(train.shape[1] - 1): # 0 to 5
                    distances.iloc[sample, neuron] += math.pow((train.iloc[sample,column] - w.iloc[neuron, column]),2)
                distances.iloc[sample, neuron] = math.sqrt(distances.iloc[sample, neuron])

        # ADJUST WEIGHTS BASED ON THE SMALLEST DISTANCE
        for line in range(distances.shape[0]):
            if (train.iloc[line, -1] - 1) == distances.iloc[line,:].idxmin():
                for column in range(w.shape[1]):
                    w.iloc[distances.iloc[line,:].idxmin(), column] += LEARNING_RATE * (train.iloc[line, column] - w.iloc[distances.iloc[line,:].idxmin(), column])
            else:
                for column in range(w.shape[1]):
                    w.iloc[distances.iloc[line,:].idxmin(), column] -= LEARNING_RATE * (train.iloc[line, column] - w.iloc[distances.iloc[line,:].idxmin(), column])

        # NORMALIZE WEIGHTS
        maximum_value = 0
        for line in range(w.shape[0]):
            for column in range(w.shape[1]):
                if(w.iloc[line,column] > maximum_value):
                    maximum_value = w.iloc[line,column]

        for line in range(w.shape[0]):
            for column in range(w.shape[1]):
                w.iloc[line, column] = w.iloc[line, column]/maximum_value

        print "[DIFF] " + str(w.values.sum() - last_w)
        
        EPOCH += 1
    print EPOCH
    return w

#////////////////////////////////////////////////////////////////////
# TEST //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def testLVQ1(w, test):
    global NUMBER_OF_CLASSES
    global LEARNING_RATE

    # EUCLIDIAN DISTANCES INITIALIZATION [NO OF TRAIN DATA, NO OF NEURONS]
    distances = pd.DataFrame(np.zeros([test.shape[0], NUMBER_OF_CLASSES]))

    # CALCULATE DISTANCES
    for sample in range(test.shape[0]): # 0 to 7
        for neuron in range(NUMBER_OF_CLASSES): # 0 to 3
            for column in range(test.shape[1]): # 0 to 5
                distances.iloc[sample, neuron] += math.pow((test.iloc[sample,column] - w.iloc[neuron, column]),2)
            distances.iloc[sample, neuron] = math.sqrt(distances.iloc[sample, neuron])

    test = pd.read_csv('./test.csv', header = None)
    for line in range(distances.shape[0]):
        print str(line + 1) + " & " + str(round(test.iloc[line,0], 4)) + " & " + str(round(test.iloc[line,1], 4)) + " & " + str(round(test.iloc[line,2], 4)) + " & " + str(round(test.iloc[line,3], 4)) + " & " + str(round(test.iloc[line,4], 4)) + " & " + str(round(test.iloc[line,5], 4)) + " & " + str(distances.iloc[line,:].idxmin()) + " \\\\"

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# CONSTANTS ---------------------------------------------------------
LEARNING_RATE = 0.05
NUMBER_OF_CLASSES = 4
NUMBER_OF_INPUTS = 6

# GET TRAIN AND TEST DATA -------------------------------------------
train = pd.read_csv('./train.csv', header = None)
test = pd.read_csv('./test.csv', header = None)

# train = pd.read_csv('./train2.csv', header = None)
# test = pd.read_csv('./test2.csv', header = None)

# NORMALIZING THE TEST AND TRAIN DATA
maximum_value = 0
for line in range(train.shape[0]):
    for column in range(train.shape[1] - 1):
        if(train.iloc[line,column] > maximum_value):
            maximum_value = train.iloc[line,column]

for line in range(test.shape[0]):
    for column in range(test.shape[1]):
        if(test.iloc[line,column] > maximum_value):
            maximum_value = test.iloc[line,column]

for line in range(train.shape[0]):
    for column in range(train.shape[1] - 1):
        train.iloc[line,column] = train.iloc[line,column]/maximum_value

for line in range(test.shape[0]):
    for column in range(test.shape[1]):
        test.iloc[line,column] = test.iloc[line,column]/maximum_value

# INITIALIZE WEIGHTS
w = pd.DataFrame(np.zeros([NUMBER_OF_CLASSES, train.shape[1] - 1]))
for line in range(train.shape[0]):
    w.iloc[train.iloc[line,6] - 1,:] = train.iloc[line,:NUMBER_OF_INPUTS]

# TRAIN LVQ-1 NETWORK
w = trainLVQ1(w, train)

# TEST LVQ-1 NETWORK
testLVQ1(w, test)