from skimage.morphology import skeletonize_3d

def skeletization(image, thresh = 150): 
    #returns the skeleton of a 3D image
    skel = image > thresh
    output = skeletonize_3d(skel)/255 
    return output

def GT_threshold(image):
    #returns the skeleton of GT 
    output = (image > 254)*1
    return output