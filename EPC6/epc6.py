# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

momentum = 0.8
learningRate = 0.1
maxEpochs = 1000
errTol = 0.000001
trainNumber = 3
numberNeurons1stLayer = 25
numberNeurons2ndLayer = 1

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def train(x, w1, w2, testNumber):
    # INITIALIZE EPOCHS AND ERR -------------------------------------
    epoch = 0
    Eqm = 1
    lEqm = 0

    xIn = x;

    # CHART LIST ---------------------------------------------------
    chart = [[],[]]
    
    while abs(Eqm - lEqm) > errTol and epoch < maxEpochs:
        lEqm = Eqm
        Eqm = 0

        lw1 = w1
        lw2 = w2

        out = []

        for t in range(0, 84):
            x = xIn.iloc[t:t+15]

            x.loc[list(x.index)[0] - 1] = -1
            x = x.sort_index()
            x = x.reset_index(drop = True)

            # INITIALIZE INPUTS
            I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
            I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
            Y1 = I1
            Y2 = I2
            D1 = pd.DataFrame(np.zeros((numberNeurons1stLayer, 1)))
            D2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer, 1)))

            # FORWARD
            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                I1.iloc[neuronIndex1stLayer,0] += (w1.iloc[:,neuronIndex1stLayer].values * x.iloc[:].values).sum()
                Y1.iloc[neuronIndex1stLayer,0] = logistic(I1.iloc[neuronIndex1stLayer,0])
            
            for name in reversed(list(Y1.index)):
                Y1.rename(index = {name : name + 1}, inplace = True)
            Y1.loc[0] = -1
            Y1 = Y1.sort_index()

            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                I2.iloc[neuronIndex2ndLayer,0] += (Y1.iloc[:,0] * w2.iloc[:,neuronIndex2ndLayer]).sum()
                Y2.iloc[neuronIndex2ndLayer,0] = logistic(I2.iloc[neuronIndex2ndLayer,0])

            # BACKWARD
            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                D2.iloc[neuronIndex2ndLayer,0] = (xIn.iloc[t + 16] - Y2.iloc[neuronIndex2ndLayer,0]) * logistic(I2.iloc[neuronIndex2ndLayer,0]) * (1 - logistic(I2.iloc[neuronIndex2ndLayer,0]))
                for wIndex2ndLayer in range(w2.shape[0]):
                    w2.iloc[wIndex2ndLayer, neuronIndex2ndLayer] += learningRate * D2.iloc[neuronIndex2ndLayer, 0] * Y1.iloc[wIndex2ndLayer, 0] + momentum * (w2.iloc[wIndex2ndLayer, neuronIndex2ndLayer] - lw2.iloc[wIndex2ndLayer, neuronIndex2ndLayer])

            for neuronIndex1stLayer in range(numberNeurons1stLayer):  # Neuronio 0, 1, 2 .... 25 de entrada   
                for neuronIndex2ndLayer in range(numberNeurons2ndLayer): # Neuronio 0 de saida
                    D1.iloc[neuronIndex1stLayer, 0] += D2.iloc[neuronIndex2ndLayer, 0] * w2.iloc[neuronIndex1stLayer, neuronIndex2ndLayer]
                D1.iloc[neuronIndex1stLayer, 0] = D1.iloc[neuronIndex1stLayer, 0] * logistic(I1.iloc[neuronIndex1stLayer,0]) * (1 - logistic(I1.iloc[neuronIndex1stLayer,0]))
                for inputIndex1stLayer in range(x.shape[0]):
                    w1.iloc[inputIndex1stLayer, neuronIndex1stLayer] += learningRate * D1.iloc[neuronIndex1stLayer,0] * x.iloc[inputIndex1stLayer] + momentum * (w1.iloc[inputIndex1stLayer, neuronIndex1stLayer] - lw1.iloc[inputIndex1stLayer, neuronIndex1stLayer])
                    
            # REPEAT FORWARD STEP

            I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
            I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
            Y1 = I1
            Y2 = I2

            # FORWARD
            for neuronIndex1stLayer in range(numberNeurons1stLayer):
                I1.iloc[neuronIndex1stLayer,0] += (w1.iloc[:,neuronIndex1stLayer].values * x.iloc[:].values).sum()
                Y1.iloc[neuronIndex1stLayer,0] = logistic(I1.iloc[neuronIndex1stLayer,0])

            for name in reversed(list(Y1.index)):
                Y1.rename(index = {name : name + 1}, inplace = True)
            Y1.loc[0] = -1
            Y1 = Y1.sort_index()

            for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
                I2.iloc[neuronIndex2ndLayer,0] += (Y1.iloc[:,0] * w2.iloc[:,neuronIndex2ndLayer]).sum()
                Y2.iloc[neuronIndex2ndLayer,0] = logistic(I2.iloc[neuronIndex2ndLayer,0])
                Eqm += 0.5 * (xIn.iloc[t + 16] - Y2.iloc[neuronIndex2ndLayer ,0]) * (xIn.iloc[t + 16] - Y2.iloc[neuronIndex2ndLayer ,0])


        # CALCULATE NEW ERROR

        Eqm = Eqm/85

        print "[EPOCH " + str(epoch) + "-> dEQM ]" + str(abs(Eqm - lEqm))

        chart[0].append(epoch)
        chart[1].append(Eqm)
        
        # INCREMENT EPOCH
        epoch += 1

    plt.figure(1)
    plt.plot(chart[0], chart[1])
    plt.savefig("./" + str(testNumber) + ".png", dpi = 500)
    plt.close()

    return w1, w2

#////////////////////////////////////////////////////////////////////
# TEST //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def test(x, w1, w2):
    xIn = x

    y = []

    # FORWARD
    for t in range(x.shape[0] - 15, x.shape[0] + 20 - 15):
        x = xIn.iloc[t:t+15]

        x.loc[list(x.index)[0] - 1] = -1
        x = x.sort_index()
        x = x.reset_index(drop = True)

        I1 = pd.DataFrame(np.zeros((numberNeurons1stLayer,1)))
        I2 = pd.DataFrame(np.zeros((numberNeurons2ndLayer,1)))
        Y1 = I1
        Y2 = I2

        # FORWARD
        for neuronIndex1stLayer in range(numberNeurons1stLayer):
            I1.iloc[neuronIndex1stLayer,0] += (w1.iloc[:,neuronIndex1stLayer].values * x.iloc[:].values).sum()
            Y1.iloc[neuronIndex1stLayer,0] = logistic(I1.iloc[neuronIndex1stLayer,0])

        for name in reversed(list(Y1.index)):
            Y1.rename(index = {name : name + 1}, inplace = True)
        Y1.loc[0] = -1
        Y1 = Y1.sort_index()

        for neuronIndex2ndLayer in range(numberNeurons2ndLayer):
            I2.iloc[neuronIndex2ndLayer,0] += (Y1.iloc[:,0] * w2.iloc[:,neuronIndex2ndLayer]).sum()
            Y2.iloc[neuronIndex2ndLayer,0] = logistic(I2.iloc[neuronIndex2ndLayer,0])
            xIn.loc[t+16] = Y2.iloc[neuronIndex2ndLayer,0]

        print xIn

    return xIn.iloc[100:120]

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


for i in range(1, trainNumber + 1):
    w1 = pd.DataFrame(np.random.rand(16, numberNeurons1stLayer))               # GENERATE WEIGHTS
    w2 = pd.DataFrame(np.random.rand(numberNeurons1stLayer, numberNeurons2ndLayer))
    x = data.iloc[:,1]                              # GET INPUTS
    print("[TRAINING NUMBER " + str(i) + "]")
    w1, w2 = train(x, w1, w2, i)
    y = test(x, w1, w2)
    print("[RESULT OF TRAINING NUMBER " + str(i) + "] " + str(y) + "\n")
