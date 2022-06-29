from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .networkSnakes import *
import time
# ===================================================================================
base = "/cvlabdata2/home/oner/CarlsData/"
# base_graphs = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/graphs_isbi12/lbl_graph/train"
path_images={"train":os.path.join(base, 'full_data'),
             "test" :os.path.join(base, 'full_data'),
             "full" :os.path.join(base, 'full_data')}
path_labels={"train":os.path.join(base, 'full_data'),
             "test" :os.path.join(base, 'full_data'),
             "full" :os.path.join(base, 'full_data')}
path_distlabels={"train":os.path.join(base, 'full_data'),
                  "test" :os.path.join(base, 'full_data'),
                  "full" :os.path.join(base, 'full_data')}
path_graphs={"train":os.path.join(base, 'full_data'),
            "test" :os.path.join(base, 'full_data'),
            "full" :os.path.join(base, 'full_data')}
train_names = [21, 139, 115, 79, 128, 82, 124, 135, 58, 44, 111, 130, 37, 108, 19]
test_names = [110, 72, 109, 75, 68, 123, 47]

sequences = {"training": train_names,
             "testing":  test_names,
             "trial":    train_names[:2],
             "all":      train_names+test_names}
s = time.time()
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels", "graph"])

def _data_point(fid, size="orig", labels="all", graph=False, threshold=15):

    basename = '6_cube_{}_new.npy'.format(fid)
    print(basename)
    filename = os.path.join(path_images[size], basename)
    image = np.float32(np.load(filename) / 255)
    
    basename = '6_lbl_{}_new.npy'.format(fid)
    filename = os.path.join(path_labels[size], basename)
    label = np.load(filename)
    
    basename = '6_distlbl_{}_new.npy'.format(fid)
    dfilename = os.path.join(path_distlabels[size], basename)
    dist_labels = np.load(dfilename)
    dist_labels = np.float32(np.clip(dist_labels, a_min=0, a_max=threshold))
    
    if graph:
        basename = '6_graph_{}.graph'.format(fid)
        gfilename = os.path.join(path_graphs[size], basename)
        g = load_graph_txt(gfilename)
    else:
        g = None

    return DataPoint(image, label, dist_labels, g)


def load_dataset(sequence='training', size="orig", labels="all", each=1, graph=False, threshold=15):

    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])

    return data_points
