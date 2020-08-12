mport numpy as np
from Perceptron import Perceptron
from PIL import Image

def buildInput(picName,num):
    X=[]
    for i in range(num):
        pic=[]
        imgOrg= Image.open("data/"+ picName+str(i)+".png")
        imgOrg.load()
        img=imgOrg.convert("1")#black&white
        img_data=np.array(img,dtype=np.uint8)

        for i in range(64):
            line=np.sum(img_data[i,:]==0)
            pic.append(line)
        X.append(pic)

    return X

def buildLabels():
    labelsS=np.zeros(10)
    labelsC=np.ones(10)
    l=[labelsC,labelsS]
    l1=np.array(l)
    l1=np.reshape(l1,(20,))
    return l1

def main():
    x1=buildInput("squere",10)
    x11= np.array(x1)
    x2=buildInput("circle",10)
    x22=np.array(x2)
    print(x11.shape,x22.shape)
    X= [x11,x22]
    XX=np.array(X)
    print(XX.shape)
    XX=np.reshape(XX,(20,64))
    labels=buildLabels()
    print(XX.shape,labels.shape)
    #print(XX)
    p=Perceptron(64)
    p.train(XX,labels)

    #test
    Xtest = buildInput("test", 8)
    xtest=np
From אסנת אנגלמן to Everyone:  11:50 AM
#test
    Xtest = buildInput("test", 8)
    xtest=np.array(Xtest)
    for i in range(8):
        prd=p.predict(Xtest[i])
        if prd==0:
            shape="circle"
        else:
            shape="squere"
        print("for picture number: ",i," - prediction =", prd,"shape=",shape,"\ntest=", Xtest)


main()
אשלח בהמשך גם את התמונות

