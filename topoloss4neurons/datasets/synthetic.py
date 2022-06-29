
from collections import namedtuple
import os
import numpy as np
from .networkSnakes import *

# ===================================================================================
image_height = -1
scales = {"orig":1}
base = "/cvlabdata2/home/oner/Snakes/codes/Synth/"
path_images={"train":os.path.join(base, 'images'),
             "test":os.path.join(base, 'images')}
path_labels={"train":os.path.join(base, 'labels'),
             "test":os.path.join(base, 'labels')}
path_distlabels={"train":os.path.join(base, 'dist_labels'),
                 "test":os.path.join(base, 'dist_labels')}
graph_labels={"train":os.path.join(base, 'graphs'),
              "test":os.path.join(base, 'graphs')}
path_labels_thin={"train":os.path.join(base, 'thin_labels')}
sequences = {"training": ["data_{}.npy".format(i) for i in range(20)],
             "testing":  ["data_{}.npy".format(i) for i in range(20,30)],
             "all":      ["data_{}.npy".format(i) for i in range(30)]}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels", "graph"])

def _data_point(fid, size="train", labels="all", graph=False, threshold=15):

    basename = fid
    print(basename)
    filename = os.path.join(path_images[size], basename)
    image = np.load(filename)

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

def load_dataset(sequence='training', size="train", labels="all", each=1, graph=False, threshold=15):

    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])

    return data_points
