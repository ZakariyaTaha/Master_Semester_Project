
from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .networkSnakes import *

# ===================================================================================
image_height = -1
scales = {"orig":1}
base = "/cvlabdata2/home/oner/Snakes/mra/MRAdata_py/"
path_images={"orig":os.path.join(base, 'img_cropped'),
             "x2":os.path.join(base, 'img_cropped_x2'),
             "test_x2":os.path.join(base, 'img_cropped_x2')}
path_labels={"orig":os.path.join(base, 'lbl_cropped'),
             "x2":os.path.join(base, 'lbl_cropped_x2'),
             "test_x2": os.path.join(base, 'lbl_cropped_x2')}
path_distlabels={"orig":os.path.join(base, 'dist_lbl_cropped'),
                 "x2":os.path.join(base, 'dist_lbl_cropped_x2'),
                 "test_x2": os.path.join(base, 'dist_lbl_cropped_x2')}
graph_labels={"orig":os.path.join(base, 'graphs'),
              "x2":os.path.join(base, 'graphs_x2'),
              "test_x2": os.path.join(base, 'test_graphs_x2')}
path_labels_thin={"orig":os.path.join(base, 'labels')}
trains = ['002.npy','003.npy','004.npy','006.npy','009.npy','010.npy','011.npy',
          '017.npy','018.npy','020.npy','021.npy','022.npy','025.npy',
          '026.npy','033.npy','037.npy','040.npy','043.npy','044.npy','045.npy',
          '047.npy','056.npy','057.npy','060.npy','064.npy','070.npy',
          '074.npy','077.npy','079.npy','082.npy','086.npy','088.npy']
tests = ['012.npy','027.npy','034.npy','054.npy','058.npy',
         '042.npy','063.npy','071.npy','023.npy','008.npy']
sequences = {"training": trains,
             "testing":  tests,
             "trial": trains[:4],
             "all":      trains + tests}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels", "graph"])

def _data_point(fid, size="orig", labels="all", graph=False, threshold=15):

    basename = '{}'.format(fid)
    print(basename)
    filename = os.path.join(path_images[size], basename)
    image = np.float32(np.load(filename))

    filename = os.path.join(path_labels[size], basename)
    label = np.load(filename)

    dfilename = os.path.join(path_distlabels[size], basename)
    dist_labels = np.load(dfilename)
    dist_labels = np.float32(np.clip(dist_labels, a_min=0, a_max=threshold))
    
    if graph:
        gfilename = os.path.join(graph_labels[size], basename[:-3]+"graph")
        g = load_graph_txt(gfilename)
    else:
        g = None

    return DataPoint(image, label, dist_labels, g)

def load_dataset(sequence='training', size="orig", labels="all", each=1, graph=False, threshold=15):

    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])

    return data_points

