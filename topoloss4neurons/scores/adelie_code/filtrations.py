import numpy as np
from skimage.morphology import binary_dilation

def thickening_3D(I_in,s=None) : #I = image in 3D matrix form (boolean) black =1, white=0 s=maximal number of steps
    #returns the thickening filtration of a 3D image
    
    I = I_in.copy() 
    if s is None:
        s = max(I.shape)  
        
    I = I_in.copy()
    
    for k in range(s): #k is the value of entry in the filtration
         
        I += binary_dilation(I)
     
    a = I.max()
    I = a - I + 1 #reverse values
    
    I = I / (s / 2) #normalize so that the highest value would be 1, obtained with only one pixel in the middle 
    
    I[I == 0] = 2 #if some values didn't enter the filtration, set to 10
        
    return I



def thickening_2D(I_in,s=None) : #I = image in matrix form (boolean) black =1, white=0, s=number of steps in the filtration
     #returns the thickening filtration of a 2D image
        
    I = I_in.copy() 
    if s is None:
        s = max(I.shape)
    
    
    I = I_in.copy()
    
    for k in range(s): #k is the value of entry in the filtration
         
        I += binary_dilation(I)
     
    a = I.max()
    I = a - I + 1 #reverse values
    
    I = I / (s / 2) #normalize so that the highest value would be 1, obtained with only one pixel in the middle 
    
    I[I == 0] = 2 #if some values didn't enter the filtration, set to 10
                
    return I




def height_2D(I_in, v) : #I_in = image in matrix form (boolean), v a vector in R^2
    #returns the height (Morse) filtration of a 2D image in the direction of v
    
    I = I_in.copy().astype(float)
    list_indices = np.where(I_in == 1)
    n = np.shape(I_in)[0]
    m = np.shape(I_in)[1]
    norm = np.linalg.norm(np.array(np.shape(I)))
    v = v / np.linalg.norm(v)
    
    if v[0] >= 0 and v [1] > 0:
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k]] = np.dot(np.array([(list_indices[1][k] + 1),n - list_indices[0][k]]),v) 

    
    if v[0] >0 and v[1] <= 0:
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k]] = np.dot(np.array([(list_indices[1][k] + 1), - list_indices[0][k] - 1 ]),v) 


    if v[0] <= 0 and v[1] < 0:
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k]] = np.dot(np.array([(list_indices[1][k] - m), - list_indices[0][k] -1 ]),v) 


    if v[0] < 0 and v[1] >= 0:
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k]] = np.dot(np.array([(list_indices[1][k] - m), (n - list_indices[0][k])]),v) 

            
    I = I / norm #normalize so that highest value of the filtration is 1. 

    I [np.where(I == 0)] = 2 # values that didn't enter the filtration are set to 10

    return I 



def height_3D(I_in,v):
     #returns the height (Morse) filtration of a 3D image in the direction of v
    
    I = I_in.copy()
    list_indices = np.where(I_in == 1)
    n = np.shape(I_in)[0]
    m = np.shape(I_in)[1]
    p = np.shape(I_in)[2]
    norm = np.linalg.norm(np.array(np.shape(I)))
    v = v / np.linalg.norm(v)
    
    if v[0] >= 0 and v[1] >=0 and v[2] >= 0 : #ppp
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(list_indices[2][k] + 1), m - list_indices[1][k], list_indices[0][k] + 1]),v) 
    
        
    if v[0] < 0 and v[1] >=0 and v[2] >= 0 : #mpp
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([- p + list_indices[2][k], m - list_indices[1][k], list_indices[0][k] + 1]),v) 
    
          
    if v[0] >= 0 and v[1] <0 and v[2] >= 0 : #pmp
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(list_indices[2][k] + 1), - (list_indices[1][k] + 1), list_indices[0][k] + 1]),v) 
    
        
    if v[0] >= 0 and v[1] >=0 and v[2] < 0 : #ppm

        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(list_indices[2][k] + 1 ), m - list_indices[1][k], - n + list_indices[0][k]]),v) 

    
    if v[0] < 0 and v[1] <0 and v[2] >= 0 : #mmp
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(-p + list_indices[2][k]), - (list_indices[1][k] + 1), list_indices[0][k] + 1]),v) 
            
    
    if v[0] < 0 and v[1] >=0 and v[2] < 0 : #mpm
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([- p + list_indices[2][k], m - list_indices[1][k], - n + list_indices[0][k]]),v) 
    
            
    if v[0] >= 0 and v[1] <0 and v[2] < 0 : #pmm
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(list_indices[2][k] + 1 ), - (list_indices[1][k] + 1), - n + list_indices[0][k]]),v) 
    
    
    if v[0] < 0 and v[1] <0 and v[2] < 0 : #mmm
        
        for k in range(len(list_indices[0])): 

            I[list_indices[0][k],list_indices[1][k],list_indices[2][k]] = np.dot(np.array([(- p + list_indices[2][k]), - (list_indices[1][k] + 1), - n + list_indices[0][k]]),v) 

            
    I = I / norm #normalize so that heighest value of filtration is 1
    I[np.where(I == 0)] = 2 # values that didn't enter the filtration are set to 10

    return I 