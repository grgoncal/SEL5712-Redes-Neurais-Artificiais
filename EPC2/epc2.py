# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#////////////////////////////////////////////////////////////////////
# CONSTANTS /////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

learningRate = 0.0025
errTol = 0.000001
trainNumber = 5
maxEpochs = 5000

#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def train(x, d, w, testNumber):
    # INITIALIZE EPOCHS AND ERR -------------------------------------
    epoch = 0
    Eqm = 1
    lEqm = 0

    # PRINT W INITIAL ----------------------------------------------
    print ("\n[W INITIAL]  " + str(w.T.values))
    
    # CHART LIST ---------------------------------------------------
    chart = [[],[]]
    
    while epoch < maxEpochs and abs(Eqm - lEqm) >= errTol:
        lEqm = Eqm
        Eqm = 0
        
        for i in range(0,x.shape[0]):
            u = w.iloc[:,0] * x.iloc[i,:]
            u = u.sum()
            w = (w.T + learningRate * (d.iloc[i] - u) * x.iloc[i,:]).T
            Eqm += (d.iloc[i] - u) * (d.iloc[i] - u)
        epoch += 1
        Eqm = Eqm/x.shape[0]
        chart[0].append(epoch)
        chart[1].append(Eqm)

    plt.figure(1)
    plt.plot(chart[0], chart[1])
    plt.savefig("./" + str(testNumber) + ".png", dpi = 500)
    plt.close()

    
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

    for i in range(0, x.shape[0]):
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
    w = pd.DataFrame(np.random.rand((data.shape[1] - 1)))
    print("[TRAINING NUMBER " + str(i) + "] ----------------------------------------------")
    y = test(train(x, d, w, i))
    print("[RESULT OF TRAINING NUMBER " + str(i) + "] " + str(y) + "\n")