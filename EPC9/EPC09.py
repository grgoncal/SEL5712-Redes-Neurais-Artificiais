# IMPORTS ----------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from PIL import Image
import random
#////////////////////////////////////////////////////////////////////
# TRAIN /////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////

def g(u):
    for i in range(u.size):
        if u[0, i] > 0:
            u[0, i] = 1
        elif u[0, i] < 0:
            u[0, i] = -1
    return u

padrao = [[] for _ in range(4)]

padrao[0] = [[-1 for _ in range(5)] for _ in range(9)]

padrao[1] = [[1 for _ in range(5)] for _ in range(9)]

padrao[2] = [[1 for _ in range(5)] for _ in range(9)]

padrao[3] = [[1 for _ in range(5)] for _ in range(9)]

for y in range(9):
    for x in range(5):
        #Desenhando 1
        if x == 2 or x == 3:
            padrao[0][y][x] = 1
        elif x == 1 and y == 1:
            padrao[0][y][x] = 1

        #Desenhando 2
        if (y==2 or y==3) and x <= 2:
            padrao[1][y][x] = -1
        if (y==5 or y==6) and x >= 2:
            padrao[1][y][x] = -1
        
        #Desenhando 3
        if (y==2 or y==3 or y == 5 or y == 6) and x <= 2:
            padrao[2][y][x] = -1
        
        #Desenhando 4
        if y >= 5 and x <= 2:
            padrao[3][y][x] = -1
        elif x==2 and y <= 2:
            padrao[3][y][x] = -1

for i in range(4):
    padrao[i] = np.matrix(padrao[i]).flatten()


#plt.imshow(padrao[3], cmap="gray")
#plt.show()

w = np.zeros((45, 45))

for i in range(4):
    w += padrao[i].T.dot(padrao[i]) - np.identity(45)
w /= 45



for iteraction in range(12):
    #20% de 45 = 9
    transm_noise = random.sample(range(45), 9)

    transm = padrao[int(iteraction/3)]
    for i in transm_noise:
        transm[0, i] = transm[0, i]*-1 

    view = transm.reshape(9, 5)
    plt.imshow(view, cmap="gray")
    plt.savefig("C:\\Users\\lavi\\Desktop\\EPC10\\" + str(iteraction) + ".jpg")
    #plt.show()

    v = transm
    err = 1

    for _ in range(50000):
        v_old = v
        u = v_old.dot(w.T)
        v = g(u)
        diff = v-v_old
        err = np.dot(diff, diff.T)
        if err < 0.000001:
            break

    transm = v.reshape(9, 5)
    plt.imshow(transm, cmap="gray")
    plt.savefig("C:\\Users\\lavi\\Desktop\\EPC10\\" + str(iteraction) + "_output.jpg")
