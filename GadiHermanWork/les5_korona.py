import numpy as np
from Perceptron import Perceptron
def ex2_korona():
    print("korona")
    inputs = np.array([ [1,0,0,1], [1,0,0,0,], [0,0,1,1,], [0,1,0,0],[1,1,0,0], [0,1,1,1], [0,0,0,1] , [0,0,1,0]])
    labels = np.array([1,1,0,0,1,1,0,0])
    p = Perceptron(4)
    p.train(inputs, labels)
    print("predict")
#    x = [1,0,1,0]
#    prd = p.predict(x)
#    print("x=", x, " , prd = ", prd)
    check(p)

def check(p):
    print("predict")
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    x = [a,b,c,d]
                    prd = p.predict(x)
                    print("x=", x, " , prd = ", prd)

ex2_korona()
