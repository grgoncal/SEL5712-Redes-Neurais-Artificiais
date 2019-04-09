# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

learningRate = 0.01
maxEpochs = 1000
errTol = 0
trainNumber = 5

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def train(x, d, w):
    # INITIALIZE EPOCHS AND ERR -------------------------------------
    epoch = 0
    err = 1

    # PRINT W INITIAL ----------------------------------------------
    print ("\n[W INITIAL]  " + str(w.T.values))

    while(epoch < maxEpochs and err != errTol):
        err = 0
        for i in range(0,x.shape[0]):
            u = w.iloc[:,0] * x.iloc[i,:]
            u = u.sum()
            if u >= 0:
                y = 1
            else:
                y = -1

            if abs(y - d.iloc[i]) == 2:
                w = (w.T + learningRate * (d.iloc[i] - y) * x.iloc[i,:]).T
                err += 1

        epoch += 1

    print ("[W FINAL]  " + str(w.T.values))
    print ("[EPOCHS]  " + str(epoch) + "\n")
    return w

#////////////////////////////////////////////////////////////////////
# TEST //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def test(w):
    x = pd.read_csv('./test.csv', header = None)
    for name in reversed(x.columns.values):
        x.rename(columns = {name : name + 1}, inplace = True)
    x.insert(loc = 0, column=0, value = np.full((x.shape[0],1), -1))

    y = []
    u = 0 

    for i in range(0, x.shape[0]):
        u = 0
        u = w.iloc[:,0] * x.iloc[i,:]
        u = u.sum()
        if u >= 0:
            y.append(1)
        else:
            y.append(-1)

    return y
    
#////////////////////////////////////////////////////////////////////
# MAIN //////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

# TRAIN DATASET ----------------------------------------------------
data = pd.read_csv('./train.csv', header = None)

# GET INPUTS, OUTPUTS AND WEIGTHS ----------------------------------
x = data.iloc[:,1:(data.shape[1] - 1)]                              # GET INPUTS
x.insert(loc = 0, column=0, value = np.full((x.shape[0],1), -1))    # ADD -1 INPUT
d = data.iloc[:,(data.shape[1] - 1)]                                # GET OUTPUTS

for i in range(1, trainNumber + 1):
    w = pd.DataFrame(np.random.rand((data.shape[1] - 1)))               # GENERATE WEIGHTS
    print("[TRAINING NUMBER " + str(i) + "] ----------------------------------------------")
    y = test(train(x, d, w))
    print("[RESULT OF TRAINING NUMBER " + str(i) + "] " + str(y) + "\n")
