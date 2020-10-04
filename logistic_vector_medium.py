 # -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:16:51 2020

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:21:02 2020

@author: maria
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.special import expit 


# Create image
folder="C:\\Users\\maria\\Desktop\\test set"
folder2="C:\\Users\\maria\\Desktop\label_images\\not"



def read_images(folder,label_val):
    '''
    Parameters
    ----------
    folder   : The folder which contains the images
    label_val: 1 --> Red pixels
              -1 --> Other pixels

    Returns
    -------
    x : IMAGE NUMPY ARRAY (Nx3)
    y : LABEL NUMPY ARRAY (Nx1)

    '''
    assert label_val==-1 or label_val==1, "Enter 1 for red / -1 for other"
    
    x=np.zeros((1,3)) #initializing the 2D data numpy array 
    y=np.ones((1,1))  #initializing the label vector    
    y=-1*y

    for filename in os.listdir(folder):
        #reading the image
        img= cv2.imread(os.path.join(folder,filename))
        
        #openCV reads in BGR format; converting to RGB
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #reshaping the 3d img matrix to 2d 
        img=np.array(np.reshape(img,(-1,img.shape[2])))
        
        #adding images to the data matrix
        x=np.vstack((x,img)) 
        
        #populating the label array
        y=np.vstack((y,label_val*np.ones((img.shape[0],1)))) 
        

            
    return x,y



def logistic_regression(x,y,iterations,alpha):
    '''
    Parameters
    ----------
    x         : IMAGE MATRIX (Nx3)
    y         : LABEL ARRAY (Nx1)
    iterations : NO. OF ITERATIONS 
    alpha     : LEARNING RATE    

    Returns
    -------
    w : WEIGHTS (3x1)

    '''
    
    w=np.zeros((3,1))
    delta = np.zeros((1, iterations))
    
    for it in range(0,iterations):
          
        product=x*(1-expit(y*(x@w)))*y
        gradient=sum(product)
        gradient=np.reshape(gradient,(gradient.shape[0],1))  # Calculating the gradient
        
        w_prev=w
        w=w+(alpha*gradient) #omega update step
        
        
        w_mag=np.linalg.norm(w,ord=2,axis=0)
        print(w_mag)
        
        w_prev_mag=np.linalg.norm(w_prev,ord=2,axis=0)
        delta[0,it]=abs(w_mag-w_prev_mag)  #calculating the absolute difference between the new omega and previous omega to check convergence.
        
        print("Iteration:", it+1)
        print("Omega",w)

    t= np.arange(1,iterations+1) 
    plt.plot(np.reshape(t, (-1, 1)),delta.T) #plotting the convergence.
    plt.show()
    
    print("Final value of Omega: ", w)
    
    return w


def segment_image(img,w):
    '''
    Parameters
    ----------
    img : TEST IMAGE
    w : WEIGHTS

    Returns
    -------
    x : BINARY MASK

    '''
    x=img.copy()
    #change color space to RGB
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB) 

    #arranging test image into Nx3 matrix
    x=np.reshape(x,(x.shape[0]*x.shape[1],x.shape[2])) 
    
    #finding the dot product <x,w>
    product=np.dot(x,w) 
    
    #classification step
    mask=np.where(product>=0,1,0) 
    
    #converting to uint8 (0-255)
    mask = mask.astype(np.uint8) 
    
    #reshaping back to img shape
    mask=np.reshape(mask,(img.shape[0],img.shape[1])) 
    
    return mask






alpha=0.0001
iterations=10

img_red,label_red=read_images(folder,1)
img_other,label_other=read_images(folder2,-1)

img_all=np.vstack((img_red,img_other))
label_all=np.vstack((label_red,label_other))
w=logistic_regression(img_all,label_all,10,0.0001)


folder = r"C:\Users\maria\Desktop\medium article\pics"

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    mask=segment_image(img,w)

plt.imshow(mask,cmap="gray")
#plt.savefig(r"C:\Users\maria\Desktop\medium article\mask.jpg",dpi=400)

