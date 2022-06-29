from scipy.linalg import norm, eigh
import numpy as np
import os
import subprocess 

def dot(x, dir):
    return 2 ** 31 if len(x) == 1 else x[0] * dir[0] + x[1] * dir[1]


def orth_proj(x, dir):
    return dot(x, dir) * dir

 #WASSERSTEIN AND SLICED WASSERSTEIN

def sliced_wasserstein(barcode1, barcode2, M, ord=1): #input : List of bars of each barcode
    """
    Approximate Sliced Wasserstein distance between two barcodes
    :param barcode1:
    :param barcode2:
    :param M: the approximation factor, bigger M means more accurate result
    :param ord: p-Wassertein distance to use
    :return:
    """
    diag = np.array([np.sqrt(2), np.sqrt(2)])
    b1 = list(barcode1)
    b2 = list(barcode2)
    for bar in barcode1:
        b2.append(orth_proj(bar, diag))
    for bar in barcode2:
        b1.append(orth_proj(bar, diag))
    b1 = np.array(b1, copy=False)
    b2 = np.array(b2, copy=False)
    s = np.pi / M
    theta = -np.pi / 2
    sw = 0
    for i in range(M):
        dir = np.array([np.cos(theta), np.sin(theta)])
        v1 = np.sort(np.dot(b1, dir))
        v2 = np.sort(np.dot(b2, dir))
        sw += s * norm(v1 - v2, ord)
        theta += s
    return sw / np.pi


#WASSERSTEIN

                  
     
def create_file(title,pd):
    #create a text file with the list of bars, compatible with wasserstein distance from https://bitbucket.org/grey_narn/hera/src/ba25f264b5d309efcf77a6b72d1b784ae97f741f/geom_matching/?at=master
    f= open(title,"w+")
    for i in range(len(pd)):
        f.write('{} {}\n'.format(pd[i][0],pd[i][1]))
    f.close()
    
    
def wasserstein(pd_1, pd_2, deg = 2, rel = 0.001, norm = 2):
    #compute wasserstein distance between two barcodes 
    
    create_file('pd_1.txt',pd_1)
    create_file('pd_2.txt',pd_2)
    
    bla = subprocess.check_output(['wasserstein_dist', 'pd_1.txt' , 'pd_2.txt', str(deg), str(rel), str(norm) ] )
    bla = float(bla.decode())
    
    os.remove('pd_1.txt')
    os.remove('pd_2.txt')
    
    return bla


 #ENTROPY

def entropy(list_pd): 
    #computes barcode's entropy
    
    p = [ list_pd[i][0] - list_pd[i][1] for i in range(len(list_pd)) ]
    L = np.sum(p)
    p = p * np.log(p/L)/L
    return - np.sum(p)
    

 #BETTI CURVES 

def betti_transform(pd) :
    
    D = []
    B = []
    for i in range(len(pd)):
        B.append(pd[i][0])
        D.append(pd[i][1])
     
    D.sort(reverse = False)
    B.sort(reverse = False)
    
    if not B:
        B = [0]
    if not D: 
        D = [0]
        
    return B,D



def betti_curve(B,D,appendZERO=False): #input is lists of birth and death values, ordered
    D = np.array(D)
    B = np.array(B)
    D = D[D != -1]
    val = np.concatenate( (np.ones(len(B)), -1*np.ones(len(D))) )
    w = np.concatenate((B,D))
    order = np.argsort(w)
    w = w[order]
    uniquew,idx,count=np.unique(w,return_index = True,return_counts = True)
    pos = idx + count - 1
    betti = np.cumsum(val[order])
    betti = betti[pos]
    if appendZERO:
        uniquew = np.append(0,uniquew)
        betti = np.append(0,betti)
    return uniquew,betti


def sub_divide(W_1, b_1,W_2,b_2): #make the same subdivision for the betti curves, to compute the distances
    #Input 
    #Two different betti curves
    #IMPORTANT W_1[0]=W_2[0]=b_1[0]=b_2[0]=0

    #Output
    W = []#finer weight grid
    B_1 = []#finner betti curve no. 1
    B_2 = []#finner betti curve no. 2
    
    
    i = 0
    j = 0
    while i < len(W_1) or j < len(W_2):
        
        if j == len(W_2) or (i < len(W_1) and j < len(W_2) and W_1[i] < W_2[j]):
            W.append(W_1[i])
            B_1.append(b_1[i])
            B_2.append(b_2[-1])
            i += 1
            
        elif i == len(W_1) or (i < len(W_1) and j < len(W_2) and W_2[j] < W_1[i]):
            W.append(W_2[j])
            B_2.append(b_2[j])
            B_1.append(b_1[-1])
            j+=1
            
        elif W_1[i] == W_2[j]:
            W.append(W_1[i])
            B_1.append(b_1[i])
            B_2.append(b_2[j])
            i += 1
            j += 1
        
    return (W, B_1, B_2)

def height_diff(W, B_1,B_2):
    #Input 
    #Two different betti curves subdivided into a common grid

    #Output
    #Sum of square of height differences
    
    W=np.array(W)
    W=np.append(W,100)
    B_1=np.array(B_1)
    B_2=np.array(B_2)
    widths=np.diff(W)
    diff_sq=np.square(np.subtract(B_1,B_2))
    height_diff=np.sum(np.multiply(widths,diff_sq))    
    return (height_diff)


def betti_dist(pd_1,pd_2):
    
    b1 = betti_transform(pd_1)
    b2 = betti_transform(pd_2)
    b1 = betti_curve(b1[0],b1[1])
    b2 = betti_curve(b2[0],b2[1])
    B = sub_divide(b1[0],b1[1],b2[0],b2[1])
    
    return height_diff(B[0],B[1],B[2])
    
# NUMBER OF DISCONNECTIONS

def number_disconnections(pd, pd_gt) :
    #input is full barcodes, not just dim 0
    return (len(pd[0]) - len(pd_gt[0]))