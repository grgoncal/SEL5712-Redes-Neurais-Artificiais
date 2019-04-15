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
        Eqm = 0

        for inputIndex in range(x.shape[0]):

            # INITIALIZE INPUTS
            I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
            I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
            Y1 = I1
            Y2 = I2
            D1 = pd.DataFrame(np.zeros((numberNeurons1stLayer, 1)))
            D2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer, 1)))
            
            # FORWARD
            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                I1.iloc[neuronIndex1stLayer] += (w1.iloc[:,neuronIndex1stLayer] * x.iloc[inputIndex,:]).sum()
                Y1.iloc[neuronIndex1stLayer] = logistic(I1.iloc[neuronIndex1stLayer])

            for name in reversed(list(Y1.index)):
                Y1.rename(index = {name : name + 1}, inplace = True)
            Y1.loc[0] = -1
            Y1 = Y1.sort_index()
            
            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                I2.iloc[neuronIndex2ndLayer] += (Y1.iloc[:,0] * w2.iloc[:,0]).sum()
                Y2.iloc[neuronIndex2ndLayer] = logistic(I2.iloc[neuronIndex2ndLayer])

            # BACKWARD
            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                D2.iloc[neuronIndex2ndLayer] = (d.iloc[inputIndex] - Y2.iloc[neuronIndex2ndLayer]) * logistic(I2.iloc[neuronIndex2ndLayer]) * (1 - logistic(I2.iloc[neuronIndex2ndLayer]))
                for wIndex2ndLayer in range(w2.shape[0]):
                    w2.iloc[wIndex2ndLayer, neuronIndex2ndLayer] += learningRate * D2.iloc[neuronIndex2ndLayer, 0] * Y1.iloc[wIndex2ndLayer, 0]

            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                    D1.iloc[neuronIndex1stLayer] = (D2.iloc[neuronIndex2ndLayer] * w2.iloc[neuronIndex1stLayer]).sum() * logistic(I1.iloc[neuronIndex1stLayer]) * (1 - logistic(I1.iloc[neuronIndex1stLayer])) 
                for inputIndex1stLayer in range(x.shape[1]):
                    w1.iloc[inputIndex1stLayer, neuronIndex1stLayer] += learningRate * D1.iloc[neuronIndex1stLayer, 0] * x.iloc[inputIndex, inputIndex1stLayer]

        # REPEAT FORWARD STEP
        for inputIndex in range(x.shape[0]):
            
            I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
            I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
            Y1 = I1
            Y2 = I2

            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                I1.iloc[neuronIndex1stLayer] += (w1.iloc[:,neuronIndex1stLayer] * x.iloc[inputIndex,:]).sum()
                Y1.iloc[neuronIndex1stLayer] = logistic(I1.iloc[neuronIndex1stLayer])

            for name in reversed(list(Y1.index)):
                Y1.rename(index = {name : name + 1}, inplace = True)
            Y1.loc[0] = -1
            Y1 = Y1.sort_index()
                
            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                I2.iloc[neuronIndex2ndLayer] += (Y1.iloc[:,0] * w2.iloc[:,0]).sum()
                Y2.iloc[neuronIndex2ndLayer] = logistic(I2.iloc[neuronIndex2ndLayer])
                Eqm += 0.5 * (d.iloc[inputIndex] - Y2.iloc[neuronIndex2ndLayer ,0]) * (d.iloc[inputIndex] - Y2.iloc[neuronIndex2ndLayer ,0])
            
        # CALCULATE NEW ERROR
        Eqm = (1/(x.shape[0] * 2)) * Eqm
        print Eqm

        # INCREMENT EPOCH
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
