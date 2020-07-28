#_____  main  ------
'''
t1 = np.array([0, 0])
print(t1 , perceptron.predict(t1))
t2 = np.array([0, 1])
print(t2 , perceptron.predict(t2))
t3 = np.array([1, 0])
print(t3 , perceptron.predict(t3))
t4 = np.array([1, 1])
print(t4 , perceptron.predict(t4))

'''
import numpy as np
from Perceptron import Perceptron

def and1():
    print("-- and --")
    p = Perceptron(2)
 #   inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
 #   labels = np.array([0, 0, 0, 1]) #  y
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
    labels = np.array([0, 0, 0, 1]) #  y
    p.train(inputs, labels)
    print("predict")
    x = [1,1]
    prd = p.predict(x)
    print("x=",x," , prd = ", prd)

def or1():
    print("-- or --")
    p = Perceptron(2)
 #   inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
 #   labels = np.array([0, 0, 0, 1]) #  y
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
    labels = np.array([0, 1, 1, 1]) #  y
    p.train(inputs, labels)
    print("predict")
    x = [1,1]
    prd = p.predict(x)
    print("x=",x," , prd = ", prd)

def xor1():
    print("-- xor --")

    p = Perceptron(2)
 #   inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
 #   labels = np.array([0, 0, 0, 1]) #  y
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # x
    labels = np.array([0, 1, 1, 0]) #  y
    p.train(inputs, labels)
    print("predict")
    x = [1,1]
    prd = p.predict(x)
    print("x=",x," , prd = ", prd)

def not1():
    print("not")
    inputs = np.array([[0],[1]])
    labels = np.array([1,0])
    p= Perceptron(1)
    p.train(inputs, labels)
    print("predict")
    x = [1]
    prd = p.predict(x)
    print("x=", x, " , prd = ", prd)
#------------------------
def ex2_korona():
    print("korona")
    inputs = [ [1,0,0,1], [1,0,0,0,], [0,0,1,1,], [0,1,0,0],[1,1,0,0], [0,1,1,1], [0,0,0,1] , [0,0,1,0]]
    labels = []
#-------    MAIN  ------
def main():
    print("check and")
    and1()
    or1()
    xor1()
    not1()

#--------------------------
main()