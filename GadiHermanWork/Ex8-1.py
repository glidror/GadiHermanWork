# Submited by Gad Lidror
# Ex 8 Part 1 - KNN
# +++++++++++++++++++++++

import numpy as np
import matplotlib.pyplot as plt
from   sklearn.metrics  import accuracy_score
from   sklearn.model_selection import cross_val_predict
import sklearn.neural_network
from   mpl_toolkits.mplot3d import Axes3D

# import self implementation of Perceptron
from Perceptron import *


# Euclidian distance between two n-dimentional points
# both ar np arrays
def Euclidian_Dist(p1,p2):
    return np.sqrt(np.sum(np.power((p1.astype(float) - p2.astype(float)), 2)))

# -------------------------------------- 
def Ex8_1_Alef():
    data = np.array( [ [ 6 , 7],
    [ 2 , 3],
    [ 3 , 7],
    [ 4 , 4],
    [ 5 , 8],
    [ 6 , 5],
    [ 7 , 9],
    [ 8 , 5],
    [ 8 , 2],
    [10 , 2] ])
    categories = np.array([0,1,1,1,1,2,2,2,2,2])
    colormap = np.array(['r', 'g', 'b'])
    plt.scatter(data[:,0], data[:,1], s=100, c=colormap[categories])
    plt.title('data - euclidean_distance')
    plt.show()

    for i in range(len(data)):
        print(Euclidian_Dist(data[0],data[i]))



def Ex8_1_Bet():
    data = np.array( [ [ 5.0 , 5.0, 5.0],
    [ 0.0 , 0.0, 0.0],
    [ 3.0 , 7.0, 2.0],
    [ 4.0 , 4.0, 8.0],
    [ 5.0 , 8.0, 9.0],
    [ 6.0 , 5.0, 7.0],
    [ 7.0 , 9.0, 4.0],
    [ 8.0 , 5.0, 1.0],
    [ 8.0 , 2.0, 3.0],
    [10.0 , 2.0, 5.0] ])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:,0], data[:,1],data[:,2], s=150)
    plt.show()

    for i in range(len(data)):
        print(Euclidian_Dist(data[0],data[i]))



def Ex8_1_Gimel(train_data, test_data, k):
    sortedInd = Train_KNN(train_data, test_data, k)

    for i in range(k):
        print (train_data[sortedInd[i]])
    print()


def Ex8_1_Dalet(train_data, test_data, train_lbl, k):
    prediction = predict(train_data,test_data,train_lbl,k)
    print('prediction:',prediction)
    print


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


def main():    
    train_data = np.array( [
    [ 2.0 , 3.0 ],
    [ 3.0 , 7.0 ],
    [ 4.0 , 4.0 ],
    [ 5.0 , 8.0 ],
    [ 6.0 , 5.0 ],
    [ 7.0 , 9.0 ],
    [ 8.0 , 5.0 ],
    [ 8.0 , 2.0 ],
    [10.0 , 2.0 ] ])

    train_lbl = np.array( [[1],[1],[1],[1],[2],[2],[2],[2],[2]])

    test_data = np.array( [[ 6.0 , 7.0 ]])

    #Ex8_1_Alef()
    #Ex8_1_Bet()
    #Ex8_1_Gimel(train_data, test_data, 1)
    #Ex8_1_Gimel(train_data, test_data, 3)
    Ex8_1_Dalet(train_data, test_data, train_lbl, 1)
    Ex8_1_Dalet(train_data, test_data, train_lbl, 3)


if __name__ == "__main__":
    main()