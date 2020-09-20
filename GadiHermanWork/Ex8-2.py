# Submited by Gad Lidror
# Ex 8 Part 2 - KNN Iris classifications
# cc
# +++++++++++++++++++++++

import  numpy as np
import  matplotlib.pyplot as plt
from    sklearn.metrics import accuracy_score
from    sklearn.metrics import confusion_matrix
from    sklearn.model_selection import cross_val_predict
import  sklearn.neural_network
from    sklearn.neighbors import KNeighborsClassifier

from    mpl_toolkits.mplot3d import Axes3D

# import self implementation of Perceptron
from Perceptron import *
 

# Euclidian distance between two n-dimentional points
# both ar np arrays
def Euclidian_Dist(p1,p2):
    return np.sqrt(np.sum(np.power((p1.astype(float) - p2.astype(float)), 2)))

def Train_KNN(train_data, test_point, k):
    distances = np.empty(len(train_data[:,0]))
    for i in range(len(train_data[:,0])):
        distances[i] = Euclidian_Dist(test_point, train_data[i])
    sortedInd = np.argsort(distances, axis = 0)
    return sortedInd

def predict (train_data, test_point, label, k):
    sortedInd = Train_KNN(train_data, test_point, k)
    LabelsRes = [label[sortedInd[i]] for i in range(k)]
    return max(LabelsRes,key=LabelsRes.count)


def Ex8_2_Alef():
    s = 120 # Size of training (from the 150 samples)
    # load the csv file into np arrays
    inputs = np.genfromtxt('Data/iris_flowers.csv', delimiter=',')
    # since the file is ordered, this is important to use ...
    np.random.shuffle(inputs)
    # already marked as 1,2,3 - no need to replace: inputs = np.where(inputs[:,-1]=='iris_setosa', 1, inputs) 

    train_data = inputs[:s,:-1]     # 120 first rows, witout type fo flower
    test_data = inputs[s:,:-1]   # remainder lines witout type fo flower
    train_lables = inputs[:s, -1].astype("int")  # lables for the test_data
    test_lables = inputs[s:,-1].astype("int")  # lables for the train_data
    return train_data, test_data, train_lables, test_lables

def Ex8_2_Bet(train_data, test_data, train_lables, test_lables):
    k = 1
    res = []
    for i in range(len(test_lables)):
        res.append (predict (train_data, test_data[i], train_lables, k))
    acc = (res == test_lables)
    return float(np.count_nonzero(acc)) / float(len(test_lables))

def Ex8_2_Gimel(train_data, test_data, train_lables, test_lables):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_data, train_lables)
    lbl_pred = classifier.predict(test_data)
    for i in range(len(lbl_pred)):
        print ("test_lables:", test_lables[i], " predictions: ", lbl_pred[i])
    print(confusion_matrix(test_lables, lbl_pred))

def Ex8_2_Dalet(train_data, test_data, train_lables, test_lables):
    mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(5), 
            solver='sgd', learning_rate_init=0.01, max_iter=1000)
    mlp.fit(train_data, train_lables)
    print(mlp.predict(train_data))
    print(mlp.score(train_data, train_lables))


def main():    
    train_data, test_data, train_lables, test_lables = Ex8_2_Alef()
    print("Alef =======================  DATA  =====================================")
    print("train_data: ", train_data)
    print("test_data: ", test_data)
    print("Alef ======================= LABELS =====================================")
    print("train_lables: ", train_lables)
    print("test_lables: ", test_lables)
    print("Bet ======================= SELF-IMPLEMENTATION PREDICT KNN =====================================")
    print (Ex8_2_Bet(train_data, test_data, train_lables, test_lables))
    print("Gimel ======================= SKLEARN_KNN =====================================")
    Ex8_2_Gimel(train_data, test_data, train_lables, test_lables)
    print("Dalet ======================= SKLEARN_NURAL_NETWORK =====================================")
    Ex8_2_Dalet(train_data, test_data, train_lables, test_lables)

if __name__ == "__main__":
    main()
