from collections import namedtuple
import os
import numpy as np
import imageio
import time
import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .. import utils

# ===================================================================================
base = "/cvlabdata2/home/oner/MALIS/RT_data/"
# base_graphs = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/graphs_isbi12/lbl_graph/train"
path_images={"train":os.path.join(base, 'imagery'),
             "test" :os.path.join(base, 'imagery_test'),
             "full" :os.path.join(base, 'full_test_images')}
path_test_images={"orig":os.path.join(base, 'imagery_test'),
             "half" :os.path.join(base, 'imagery_test')}
path_labels={"orig":os.path.join(base, 'masks_thick'),
             "half" :os.path.join(base, 'masks_thick')}
path_labels_thin={"orig":os.path.join(base, 'masks'),
                  "half" :os.path.join(base, 'masks')}
path_labels_dist={"train":os.path.join(base, 'dist_lbl'),
                  "test" :os.path.join(base, 'dist_lbl')}
train_names = [x for x in os.listdir(os.path.join(base, path_images["train"])) if not x.startswith(".")]
test_names = [x for x in os.listdir(os.path.join(base, path_images["test"])) if not x.startswith(".")]
full_test_names = sorted([x for x in os.listdir(os.path.join(base, path_images["full"])) if (not x.startswith(".") and x.endswith("fixed.png"))])
sequences = {"training": train_names,
             "testing":  test_names,
             "trial":    train_names[:20],
             "all":      train_names+test_names,
             "full":     full_test_names}
s = time.time()
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "weights", "basename", "filename", "node_coords", "edges"])

def _data_point(fid, size="train", labels="all", dist_lbl=True, graph=False, dist_thresh=20):
    s = time.time()
    basename = fid
    filename = os.path.join(path_images[size], basename)
    print(filename)
    image = imageio.imread(filename)
    weights = None
    # if labels=="orig" or labels=="all":
    #     filename = os.path.join(path_labels[size], basename)
    #     label = imageio.imread(filename[:-7] + "osm.png")
    #     if dist_lbl:
    #         label = dist(1 - label//255)
    #     else:
    #         label = np.uint8(label<1) # inverting background-foreground
    # else:
    label = None

    # if labels=="thin" or labels=="all":

    if labels == None:
        label_thin = None
        filename = None
    else:
        filename = os.path.join(path_labels_dist[size], basename)
        label_thin = tifffile.imread(filename[:-7] + "osm.tif")
        label_thin[label_thin > dist_thresh] = dist_thresh
    # weights = np.ones((label_thin.shape[0], label_thin.shape[1]))
        # weights = 2*dist_thresh - label_thin
        # weights /= weights.mean()

    node_coords = None
    edges = None
    print(time.time() - s)
    return DataPoint(image, label, label_thin, weights, basename, filename, node_coords, edges)

def load_dataset(sequence='training', size="train", labels="all", each=1, dist_lbl=True, graph=False, dist_thresh=np.inf):

    data_points = tuple(_data_point(fid, size, labels, dist_lbl, graph, dist_thresh) for fid in sequences[sequence][::each])

    return data_points
