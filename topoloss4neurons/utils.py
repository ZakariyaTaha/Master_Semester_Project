import numpy as np
import pickle

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data
        
def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f) 

def thick_label_gaussian_weights(label, class_weights, sigma=6, label_th=0.05):
    import cv2
    
    assert label.min()>=0 and label.max()<=1
    assert np.ndim(label)==2
    
    h,w = label.shape[:2]
    
    mag_neg, mag_pos = class_weights
    diff = mag_pos-mag_neg
    
    weights = np.float32(label)*diff
    weights = cv2.GaussianBlur(weights, ksize=None, sigmaX=sigma)

    thick_label = (weights/diff)>label_th

    weights += np.ones((h,w))*mag_neg
    
    # do we need to mask the weight using he thick_label?
    
    return thick_label, weights