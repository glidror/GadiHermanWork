import numpy as np
from Perceptron import Perceptron
import matplotlib.pyplot as plt 
from PIL import Image  
import time 

def GetColorAverage(path, num):
    colors = []
    for i in range(num):
        img = Image.open(path + str(i) + ".jpg") 
        img.load() 
        data = np.array(img, dtype=np.uint8)
        colors.append([np.mean(data[:,:,0]), np.mean(data[:,:,1]), np.mean(data[:,:,2])])
    return colors

def PrintSpectrum(d_array):
    plt.subplot(1,3,1)
    plt.scatter(d_array[:10,0], d_array[:10,1], color='r')
    plt.scatter(d_array[10:,0], d_array[10:,1], color='g')
    plt.title('Red and Green')
    plt.subplot(1,3,2)
    plt.scatter(d_array[:10,0], d_array[:10,2], color='r')
    plt.scatter(d_array[10:,0], d_array[10:,2], color='b')
    plt.title('Red and Blue')
    plt.subplot(1,3,3)
    plt.scatter(d_array[:10,1], d_array[:10,2], color='g')
    plt.scatter(d_array[10:,1], d_array[10:,2], color='b')
    plt.title('Green and Blue')
    plt.show()

def ex3_see_land():
    p = Perceptron(3)
    classes = np.array([b'Sea', b'Land']) # the list of classes

    sea_colors = GetColorAverage('Data/data/sea', 10)
    land_colors = GetColorAverage('Data/data/land', 10)
    
    data_colors =  sea_colors + land_colors   
    d_array = np.array(data_colors)
    d_label = [0]*10 + [1]*10

    PrintSpectrum(d_array)
    
    
    p.train(d_array, d_label)

    test_colors = GetColorAverage('Data/test/test', 6)
    
    t_array = np.array(test_colors)
    t_label = [0,1,0,1,1,1]
    
    
    for i in range(len(t_label)):
        print (f"Predict picture is: {classes[p.predict(t_array[i])]} , actual:{classes[t_label[i]]}")
      



plt.rcParams['figure.figsize'] = (5.0, 5.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

ex3_see_land()
