# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

learningRate = 0.1
maxEpochs = 10000
errTol = 0.000001
trainNumber = 5
numberNeurons1stLayer = 10
numberNeurons2ndLayer = 1

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def train(x, d, w1, w2):
    # INITIALIZE EPOCHS AND ERR -------------------------------------
    epoch = 0
    Eqm = 1
    lEqm = 0

    while abs(Eqm - lEqm) > errTol and epoch < maxEpochs:
        lEqm = Eqm
    
        for inputIndex in range(x.shape[0]):

            # INITIALIZE INPUTS
            I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
            I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
            Y1 = I1
            Y2 = I2
            # D1 = pd.DataFrame(np.zeros((w1.shape[0],w1.shape[1])))
            D2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,numberNeurons2ndLayer)))

            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                I1.iloc[neuronIndex1stLayer] += (w1.iloc[:,neuronIndex1stLayer] * x.iloc[inputIndex,:]).sum()
                Y1.iloc[neuronIndex1stLayer] = logistic(I1.iloc[neuronIndex1stLayer])
                for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                    I2.iloc[neuronIndex2ndLayer] += (Y1.iloc[:,0] * w2.iloc[:,0]).sum()
                    Y2.iloc[neuronIndex2ndLayer] = logistic(I2.iloc[neuronIndex2ndLayer])

            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                D2.iloc[neuronIndex2ndLayer] = (d.iloc[inputIndex] - Y2.iloc[neuronIndex2ndLayer]) * logistic(I2.iloc[neuronIndex2ndLayer]) * (1 - logistic(I2.iloc[neuronIndex2ndLayer]))
                for wIndex2ndLayer in range(w2.shape[0] - 1):
                    w2.iloc[wIndex2ndLayer, neuronIndex2ndLayer] += learningRate * D2.iloc[neuronIndex2ndLayer, 0] * Y1.iloc[wIndex2ndLayer, 0]
                    # print "[LOG]" + str(w2.iloc[wIndex2ndLayer, neuronIndex2ndLayer]) + " and " + str(D2.iloc[neuronIndex2ndLayer, 0]) + " and " + str(Y1.iloc[wIndex2ndLayer,0])
            print w2

        epoch += 1

    return w1, w2

#////////////////////////////////////////////////////////////////////
# TEST //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def test(w):
    return 0

#////////////////////////////////////////////////////////////////////
# LOGISTIC FUNCTION /////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def logistic(x):
    return 1 / (1 + math.exp(-x))

#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# TRAIN DATASET ----------------------------------------------------
data = pd.read_csv('./train.csv', header = None)

# GET INPUTS, OUTPUTS AND WEIGTHS ----------------------------------
x = data.iloc[:,1:(data.shape[1] - 1)]                              # GET INPUTS
x.insert(loc = 0, column=0, value = np.full((x.shape[0],1), -1))    # ADD -1 INPUT

d = data.iloc[:,(data.shape[1] - 1)]     

for i in range(1, trainNumber + 1):
    w1 = pd.DataFrame(np.random.rand((data.shape[1] - 1), numberNeurons1stLayer))               # GENERATE WEIGHTS
    w2 = pd.DataFrame(np.random.rand(numberNeurons1stLayer + 1, numberNeurons2ndLayer))

    print("[TRAINING NUMBER " + str(i) + "]")
    w1, w2 = train(x, d, w1, w2)
    # y = test()
    # print("[RESULT OF TRAINING NUMBER " + str(i) + "] " + str(y) + "\n")
