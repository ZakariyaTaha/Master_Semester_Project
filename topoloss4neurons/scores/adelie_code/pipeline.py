#distances 
import preprocessing as pre
import barcodes as bar
import filtrations as filt
import distances as dist
import pickle
import numpy as np


def score_wasserstein_thickening_3D(image, image_gt):
    
    #preprocessing
    
    image_s = pre.skeletization(image)
    image_s_gt = pre.GT_threshold(image_gt)
    
    #save them 
    np.save('image_s',image_s)
    np.save('image_s_gt',image_s_gt)
    
    #filtration
    
    f = filt.thickening_3D(image_s)
    fgt = filt.thickening_3D(image_s_gt)
        
    
    #save filtration
    np.save('filt',f)
    
    #computing persistence
    
    pd_vis = bar.barcode(f) #for visualization
    pd = bar.separate_barcode(pd_vis) #list of dim 0,1 and 2

    pd_vis_gt = bar.barcode(fgt) #for visualization
    pd_gt = bar.separate_barcode(pd_vis_gt) #list of dim 0,1 and 2
    
    #save them 
    f = open('pd_vis','wb')
    pickle.dump(pd_vis,f)
    f.close() 

    f = open('pd_vis_gt','wb')
    pickle.dump(pd_vis_gt,f)
    f.close() 
    
    f = open('pd','wb')
    pickle.dump(pd,f)
    f.close() 
    
    f = open('pd_gt','wb')
    pickle.dump(pd_gt,f)
    f.close() 
    
    #compute distances
    d0 = dist.wasserstein(pd[0], pd_gt[0])
    d1 = dist.wasserstein(pd[1], pd_gt[1])
    d2 = dist.wasserstein(pd[2], pd_gt[2])
    

    return d0, d1, d2


def score_wasserstein_thickening_2D(image, image_gt):
    
    #preprocessing
    
    #image_s = pre.skeletization(image)
    #image_s_gt = pre.GT_threshold(image_gt)
    
    #save them 
    #np.save('image_s',image_s)
    #np.save('image_s_gt',image_s_gt)
    
    #filtration
    
   
    f = filt.thickening_2D(image)
    fgt = filt.thickening_2D(image_gt)
        
    
    #save filtration
    np.save('filt',f)
    
    #computing persistence
    
    pd_vis = bar.barcode(f) #for visualization
    pd = bar.separate_barcode(pd_vis) #list of dim 0,1 and 2

    pd_vis_gt = bar.barcode(fgt) #for visualization
    pd_gt = bar.separate_barcode(pd_vis_gt) #list of dim 0,1 and 2
    
    #save them 
    f = open('pd_vis','wb')
    pickle.dump(pd_vis,f)
    f.close() 

    f = open('pd_vis_gt','wb')
    pickle.dump(pd_vis_gt,f)
    f.close() 
    
    f = open('pd','wb')
    pickle.dump(pd,f)
    f.close() 
    
    f = open('pd_gt','wb')
    pickle.dump(pd_gt,f)
    f.close() 
    
    
    #compute distances
    d0 = dist.wasserstein(pd[0], pd_gt[0])
    d1 = dist.wasserstein(pd[1], pd_gt[1])
    

    return d0, d1


def score_wasserstein_height_2D(image, image_gt, vec):
    
    f = filt.height_2D(image, vec)
    fgt = filt.height_2D(image_gt,vec)
        
    
    #save filtration
    np.save('filt',f)
    
    #computing persistence
    
    pd_vis = bar.barcode(f) #for visualization
    pd = bar.separate_barcode(pd_vis) #list of dim 0,1 and 2

    pd_vis_gt = bar.barcode(fgt) #for visualization
    pd_gt = bar.separate_barcode(pd_vis_gt) #list of dim 0,1 and 2
    
    #save them 
    f = open('pd_vis','wb')
    pickle.dump(pd_vis,f)
    f.close() 

    f = open('pd_vis_gt','wb')
    pickle.dump(pd_vis_gt,f)
    f.close() 
    
    f = open('pd','wb')
    pickle.dump(pd,f)
    f.close() 
    
    f = open('pd_gt','wb')
    pickle.dump(pd_gt,f)
    f.close() 
 
    
    #compute distances
    d0 = dist.wasserstein(pd[0], pd_gt[0])
    d1 = dist.wasserstein(pd[1], pd_gt[1])
    
    return d0, d1


def score_wasserstein_height_3D(image, image_gt,vec):
    
    f = filt.height_3D(image,vec)
    fgt = filt.height_3D(image_gt,vec)
        
    
    #save filtration
    np.save('filt',f)
    
    #computing persistence
    
    pd_vis = bar.barcode(f) #for visualization
    pd = bar.separate_barcode(pd_vis) #list of dim 0,1 and 2

    pd_vis_gt = bar.barcode(fgt) #for visualization
    pd_gt = bar.separate_barcode(pd_vis_gt) #list of dim 0,1 and 2
    
    #save them 
    f = open('pd_vis','wb')
    pickle.dump(pd_vis,f)
    f.close() 

    f = open('pd_vis_gt','wb')
    pickle.dump(pd_vis_gt,f)
    f.close() 
    
    f = open('pd','wb')
    pickle.dump(pd,f)
    f.close() 
    
    f = open('pd_gt','wb')
    pickle.dump(pd_gt,f)
    f.close() 
    
    
    #compute distances
    d0 = dist.wasserstein(pd[0], pd_gt[0])
    d1 = dist.wasserstein(pd[1], pd_gt[1])
    return d0, d1