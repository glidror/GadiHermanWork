import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image  
import time 


# Lesson 1 class Ex 1
def L1_Ex1():
    x = np.zeros(30)
    y1 = np.zeros(30)
    y2 = np.zeros(30)

    for i in range(0, 30): 
        x[i] = i 
        y1[i] = i ** 2 
        y2[i] = i ** 3 
    print(x) 
    print(y1) 
    print(y2) 
    plt.plot(x, y1, 'r--') 
    plt.plot(x, y2, 'g^') 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.yscale('log') 
    plt.show() 

# Lesson 1 class Ex 2
def L1_Ex2():
    x = np.arange(-30,30)
    y = np.zeros(60)
    y = x**2
    print(x) 
    print(y) 
    plt.plot(x, y, 'r') 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.show() 

# Lesson 1 class Ex 3
def L1_Ex3():
    t = np.arange(0,0.004,0.00001)
    y1 = 5 + np.sin(2*np.pi*500*t)
    y2 = 5 * np.sin(2*np.pi*1000*t)

    plt.plot(t, y1) 
    plt.plot(t, y2) 
    plt.xlabel('t') 
    plt.ylabel('y') 
    plt.show() 


# Lesson 1 class Ex 4
def L1_Ex4():
    x = np.arange(-20,20)
    y1 = 10*x + 6
    y2 = 2*x**2+2*x-100
    plt.plot(x, y1) 
    plt.plot(x, y2) 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.show() 

# Lesson 1 class Ex 5
def L1_Ex5():
    sum=0 
    x1 = []
    x2 = np.zeros(1000000)

    for i in range(1000000): 
        x1.append(i)
    for i in range(1000000): 
        x2[i] = i

    print ("List")
    startTime = time.time() 

    for i in range(len(x1)):
        x1[i] *= 5

    endTime = time.time() 
    print("Time elapsed:" , (endTime - startTime) , "seconds") 

    print ("NumPy")
    startTime = time.time() 
    x2 *= 5
    endTime = time.time() 
    print("Time elapsed:" , (endTime - startTime) , "seconds") 

    print (x1[1])
    print (x2[1])

# Lesson 1 class Ex 6
def L1_Ex6():
    size = 5
    m = np.ones((5,5))
    print (m)
    print()

    m[1:-1,1:-1] = 0
    print ( m )

# Lesson 1 class Ex 7
def L1_Ex7():
    a1 = np.random.randint(10,size=[5])
    a2 = np.random.randint(10,size=[7])
    a3 = []
    for x in a1:
        if (x in a2):
            a3.append(x)

    print (a1)
    print (a2)
    print (a3)

# Lesson 1 class Ex 8
def L1_Ex8():
    a = np.array([1,3, 11,12,21,22,2,4,  3, 14, 1, 24])

    a1 = a.reshape(2,3,2)
    print (a)

    print(a1)
    print(a1*2)
    print(a1+10)

# Lesson 1 class Ex 9
def L1_Ex9():
    x1 = np.array([ [1] , [2] , [3] , [4] , [5 ], [6 ] ]) 
    x2 = np.array([ [7] , [8] , [9] , [10] ,[11], [12] ]) 
    x3 = np.array([ [13], [14], [15], [16], [17], [18] ]) 

    x_all = np.concatenate((x1,x2, x3), axis =0)

    print (x_all)
    xa = x_all.reshape(3,len(x1))
    print()
    xb = [] #np.array(len(x_all))
    for i in range(len(x1)):
        xb.append (xa[:,i] )
    xb = np.array(xb)
    print (xb)
    print()
    print (xb[:3])
    print()
    x1 = np.array([ [1] , [2] , [3] , [4] , [5 ], [6 ] ]) 
    x2 = np.array([ [7] , [8] , [9] , [10] ,[11], [12] ]) 
    x3 = np.array([ [13], [14], [15], [16], [17], [18] ]) 
    x_m = np.array([x1,x2,x3])

    print(x_m)
    xc = x_all.reshape(2,3,3)
    print()
    print (xc)

# Lesson 1 class Ex 10
def L1_Ex10():
    pic = np.ndarray(shape=(100,100,3), dtype = np.uint8)
    pic[0:50, 0:50] = [255, 0, 0]
    pic[0:50, -50:] = [0, 255, 0]
    pic[-50:, 0:50] = [0, 0, 255]
    pic[-50:, -50:] = [255, 255, 255]

    plt.imshow(pic, interpolation='nearest') 
    plt.show() 


# Lesson 1 class Ex 11
def L1_Ex11():
    # plain picture
    img = Image.open("data/img1.jpg") 
    img.load() 
    img.show() 

    # Gray scale
    # weighted average for nice gray-like results 
    data = np.array(img,dtype=np.uint8) 
    data=data[:,:,0]*0.3+data[:,:,1]*0.59+data[:,:,2]*0.11
    img1 = Image.fromarray(data) 
    img1.show()  

    # Black and White
    # Turn into boolean values (1,0)
    data=data>128 
    img2 = Image.fromarray(data) 
    img2.show() 

# Lesson 1 class Ex 12
def L1_Ex12():
    pic = np.ndarray(shape=(100,200,3), dtype = np.uint8)
    pic[:, :] = [255, 0, 0]
    pic[25:75, 25:75] = [0, 0, 255]
    pic[:25, 125:175] = [0, 0, 255]
    pic[75:, 125:175] = [0, 0, 255]
    pic[:, 100:125] = [0, 0, 255]
    pic[:, 175:] = [0, 0, 255]

    plt.imshow(pic, interpolation='nearest') 
    plt.show() 


L1_Ex1()
L1_Ex2()
L1_Ex3()
L1_Ex4()
L1_Ex5()
L1_Ex6()
L1_Ex7()
L1_Ex8()
L1_Ex9()
L1_Ex10()
L1_Ex11()
L1_Ex12()

