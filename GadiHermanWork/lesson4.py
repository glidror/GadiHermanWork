import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image  
import time 



x = np.array([23,26,30,34,43,48,52,57,58]) 
y = np.array([651,762,856,1063,1190,1298,1421,1440,1518])


def GradientDescent(x,y,learning_rate=0.01, epochs=20):
    m=0
    b=0
    m_list = []
    for _ in range(epochs):
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            guess = m * xi + b
            error = guess - yi
            m = m - (error * xi) * learning_rate
            m_list.append(m)
            b = b - error * learning_rate
    return m,b, m_list


m,b, m_list = GradientDescent(x,y)
m_list = np.array(m_list)

print (m,b)
plt.plot(m_list)
#plt.scatter(x,y)
#plt.plot(x,m*x+b)
plt.show()
