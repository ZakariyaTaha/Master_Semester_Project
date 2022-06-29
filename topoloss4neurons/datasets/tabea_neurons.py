
from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .networkSnakes import *

# ===================================================================================
image_height = -1
scales = {"orig":1} 
base = "/cvlabsrc1/home/oner/tabea_dataset/"
path_images={"train":os.path.join(base, 'images'),
             "test":os.path.join(base, 'test_image'),
             "video":os.path.join(base, 'video_image')}
path_labels={"train":os.path.join(base, 'labels'),
             "test":os.path.join(base, 'test_label'),
             "video":os.path.join(base, 'video_label')}
path_dist_labels={"train":os.path.join(base, 'dist_labels'),
             "test":os.path.join(base, 'test_dist_label'),
             "video":os.path.join(base, 'video_dist_label')}
path_graphs={"train":os.path.join(base, 'graphs'),
             "test":os.path.join(base, 'test_graph'),
             "video":os.path.join(base, 'video_graph')}
path_labels_thin={"train":os.path.join(base, 'labels')}
sequences = {"training": [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,0],[3,1],[3,2]],
             "testing":  [[0,0]],
             "unlabeled":[],
             "all":      ["2","3"]}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels", "graph"])

def _data_point(fid, size="train", labels="all", graph=False, threshold=15):
    
    basename = 'image_{}_{}.npy'.format(fid[0],fid[1])
    print(basename)
    filename = os.path.join(path_images[size], basename)
    image = np.float32(np.load(filename)/214)
    
    basename = 'label_{}_{}.npy'.format(fid[0],fid[1])
    filename = os.path.join(path_labels[size], basename)
    label = np.load(filename)
    
    basename = 'dist_label_{}_{}.npy'.format(fid[0],fid[1])
    filename = os.path.join(path_dist_labels[size], basename)
    dlabel = np.float32(np.load(filename))
    
    dlabel[dlabel > threshold] = threshold
    
    if graph:
        basename = 'graph_{}_{}.graph'.format(fid[0],fid[1])
        gfilename = os.path.join(path_graphs[size], basename)
        g = load_graph_txt(gfilename)
    else:
        g = None

    return DataPoint(image, label, dlabel, g)
        
def load_dataset(sequence='training', size="train", labels="all", each=1, graph=False, threshold=15):
    
    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])
        
    return data_points
