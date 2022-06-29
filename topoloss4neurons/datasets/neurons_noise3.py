
from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .networkSnakes import *

# ===================================================================================
image_height = -1
scales = {"orig":1}
base = "/cvlabdata2/home/oner/snakesRegression/"
graph_base = "/cvlabdata2/home/oner/Snakes/brain/"
path_images={"train":os.path.join(base, 'images'),
             "test":os.path.join(base, 'images')}
path_labels={"train":os.path.join(base, 'labels'),
             "test":os.path.join(base, 'labels')}
path_distlabels={"train":os.path.join(base, 'noise3_dist_labels'),
                 "test":os.path.join(base, 'dist_labels')}
graph_labels={"train":os.path.join(base, 'noise3_graphs'),
              "test":os.path.join(graph_base, 'graphs_old')}
path_labels_thin={"orig":os.path.join(base, 'labels')}
sequences = {"training": ['6.t7.npy','14.t7.npy',
                         '17.t7.npy','3.t7.npy',
                         '5.t7.npy','0.t7.npy',
                         '12.t7.npy','13.t7.npy',
                         '1.t7.npy','8.t7.npy'],
             "testing":  ['11.t7.npy', '4.t7.npy',
                          '10.t7.npy', '16.t7.npy'],
             "all":      ['6.t7.npy','14.t7.npy',
                         '17.t7.npy','3.t7.npy',
                         '5.t7.npy','0.t7.npy',
                         '12.t7.npy','13.t7.npy',
                         '1.t7.npy','8.t7.npy',
                          '11.t7.npy', '4.t7.npy',
                          '10.t7.npy', '16.t7.npy']}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels", "graph"])

def _data_point(fid, size="train", labels="all", graph=False, threshold=15):

    basename = '{}'.format(fid)
    print(basename)
    filename = os.path.join(path_images[size], basename)
    image = np.load(filename)[0,:,:,:]
    image /= image.max()

    filename = os.path.join(path_labels[size], basename)
    label = np.load(filename)

    dfilename = os.path.join(path_distlabels[size], basename)
    dist_labels = np.load(dfilename)
    dist_labels[dist_labels > threshold] = threshold
    
    if graph:
        gfilename = os.path.join(graph_labels[size], basename[:-3]+"graph")
        g = load_graph_txt(gfilename)
    else:
        g = None

    return DataPoint(image, label, dist_labels, g)

def load_dataset(sequence='training', size="orig", labels="all", each=1, graph=False, threshold=15):

    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])

    return data_points
