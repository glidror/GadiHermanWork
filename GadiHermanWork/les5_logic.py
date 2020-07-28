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
    p = Perceptron(2)
    p.train(inputs, labels_and)
    print("predict and")
    prd_and = p.predict([1,0])
    print (prd_and)

    p = Perceptron(2)
    p.train(inputs, labels_or)
    print("predict or")
    prd_or = p.predict([1,0])
    print (prd_or)

    p = Perceptron(2)
    p.train(inputs, labels_xor)
    print("predict xor")
    prd_xor = p.predict([1,0])
    print (prd_xor)

    p = Perceptron(1)
    p.train(inputs_not, labels_not)
    print("predict not")
    prd_not = p.predict([0])
    print (prd_not)

ex2_Logic()
