
from collections import namedtuple
import os
import numpy as np
import imageio
from scipy.ndimage.morphology import distance_transform_edt as dist
from .. import utils
import pickle

# ===================================================================================
image_height = 512
scales = {"orig":1,
          "256" :256/image_height}
base = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/orig"
base_graphs = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/graphs_isbi12/lbl_graph/train"
base_ns = "/cvlabdata2/home/oner/Snakes/NetworkSnakes/"
path_images={"orig":os.path.join(base, 'train_images'),
             "256" :os.path.join(base, 'train_images_256')}
path_labels={"orig":os.path.join(base, 'train_labels'),
             "256" :os.path.join(base, 'train_labels_256')}
path_labels_thin={"orig":os.path.join(base, 'train_labels_thin'),
                  "256" :os.path.join(base, 'train_labels_thin_256')}
sequences = {"training": range( 0,15,1),
             "testing":  range(15,30,1),
             "all":      range( 0,30,1)}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "weights", "basename", "filename", "node_coords", "edges", "ns"])

def _data_point(fid, size="orig", labels="all", dist_lbl=True, graph=False, dist_thresh=20, snakes=True):

    basename = 'slice_{:02d}.tiff'.format(fid)
    filename = os.path.join(path_images[size], basename)
    image = imageio.imread(filename)
    weights = None
    if labels=="orig" or labels=="all":
        filename = os.path.join(path_labels[size], basename)
        label = imageio.imread(filename)
        if dist_lbl:
            label = dist(label)
        else:
            label = np.uint8(label<1) # inverting background-foreground
    else:
        label = None

    if labels=="thin" or labels=="all":
        filename = os.path.join(path_labels_thin[size], basename)
        label_thin = imageio.imread(filename)
        if dist_lbl:
            label_thin = dist(label_thin)
            label_thin[label_thin > dist_thresh] = dist_thresh
            # weights = np.ones((label_thin.shape[0], label_thin.shape[1]))
            # weights = 2*dist_thresh - label_thin
            # weights /= weights.mean()
        else:
            label_thin = np.uint8(label_thin<1) # inverting background-foreground
    else:
        label_thin = None

    weights = None

    if graph:
        filename = os.path.join(base_graphs, 'slice_{:02d}.pickle'.format(fid))
        data = utils.pickle_read(filename)
        node_coords = data['node_coordinates']
        edges = data['edges']
    else:
        node_coords = None
        edges = None

    if snakes:
        with open(os.path.join(base_ns, 'slice_{:02d}.ns'.format(fid)), 'rb') as network_s:
            ns = pickle.load(network_s)
    else:
        ns = None

    return DataPoint(image, label, label_thin, weights, basename, filename, node_coords, edges, ns)

def load_dataset(sequence='training', size="orig", labels="all", each=1, dist_lbl=True, graph=False, dist_thresh=20, snakes=True):

    data_points = tuple(_data_point(fid, size, labels, dist_lbl, graph, dist_thresh, snakes) for fid in sequences[sequence][::each])

    return data_points
