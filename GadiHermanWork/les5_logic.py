import numpy as np
from Perceptron import Perceptron


def ex2_Logic():
    print("Logic")
    inputs = np.array([ [0,0], [1,0], [0,1], [1,1]])
    labels_and = np.array([0,0,0,1])
    labels_or = np.array([0,1,1,1])
    labels_xor = np.array([0,1,1,0])
    inputs_not = np.array([0,1])
    labels_not = np.array([1,0])


    print("predict and")
    p = Perceptron(2, epochs=2000, learningRate=0.001)
    p.train(inputs, labels_and)
    for i in range (2):
        for j in range (2):
            prd = p.predict([i,j])
            print("x=", i, j, " , prd = ", prd)

    print("predict xor")
    p = Perceptron(2, epochs=2000, learningRate=0.001)
    p.train(inputs, labels_xor)
    for i in range (2):
        for j in range (2):
            prd = p.predict([i,j])
            print("x=", i, j, " , prd = ", prd)
    
    print("predict or")
    p = Perceptron(2, epochs=2000, learningRate=0.001)
    p.train(inputs, labels_or)
    for i in range (2):
        for j in range (2):
            prd = p.predict([i,j])
            print("x=", i, j, " , prd = ", prd)


    p = Perceptron(1)
    p.train(inputs_not, labels_not)
    print("predict not")
    prd_not = p.predict([0])
    print ("not 0 is ", prd_not)
    prd_not = p.predict([1])
    print ("not 1 is ", prd_not)
    
ex2_Logic()
