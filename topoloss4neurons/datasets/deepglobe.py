from collections import namedtuple
import os
import numpy as np
import imageio
import time
import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .. import utils

# ===================================================================================
# base = "/cvlabsrc1/cvlab/datasets_leo/deep_globe_cvpr_2018/dataset/train"
# base2 = "/cvlabsrc1/cvlab/datasets_leo/deep_globe_cvpr_2018/dataset/road-train-2+valid.v2/train"
base = "/cvlabdata2/home/oner/MALIS/road_connectivity/data/deepglobe"
base_gt = "/cvlabsrc1/home/oner/DeepGlobeDistLabels"
path_images={"train":os.path.join(base, 'train', 'images'),
             "test" :os.path.join(base, 'val', 'images')}
path_test_images={"orig":os.path.join(base, 'imagery_test'),
             "half" :os.path.join(base, 'imagery_test')}
path_labels={"orig":os.path.join(base, 'masks_thick'),
             "half" :os.path.join(base, 'masks_thick')}
path_labels_thin={"orig":os.path.join(base, 'masks'),
                  "half" :os.path.join(base, 'masks')}
path_labels_dist={"train":os.path.join(base, 'dist_lbl'),
                  "test" :os.path.join(base, 'dist_lbl')}

# names = np.array([os.path.join(base, x) for x in os.listdir(base) if x.endswith(".jpg")] + [os.path.join(base2, x) for x in os.listdir(base2) if x.endswith(".jpg")])

train_names = np.array([os.path.join(path_images['train'], x) for x in os.listdir(path_images['train']) if x.endswith(".jpg")])
test_names = np.array([os.path.join(path_images['test'], x) for x in os.listdir(path_images['test']) if x.endswith(".jpg")])

# indexes = np.arange(len(names))
# np.random.seed(0)
# np.random.shuffle(indexes)
#
# train_names = names[indexes[:4996]]
# test_names = names[indexes[4996:]]

sequences = {"training": train_names,
             "testing":  test_names,
             "trial": train_names[:16],
             "all":      train_names}
s = time.time()
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "basename", "filename"])

def _data_point(fid, size="train", labels="all", dist_lbl=True, graph=False, dist_thresh=20):
    s = time.time()
    basename = fid
    filename = basename
    image = imageio.imread(filename)

    label = None

    label_dist = tifffile.imread(os.path.join(base_gt, filename.split("/")[-1][:-7] + "mask.png"))
    label_dist[label_dist > dist_thresh] = dist_thresh
    print(time.time() - s)
    return DataPoint(image, label, label_dist, basename, filename)

def load_dataset(sequence='training', size="train", labels="all", each=1, dist_lbl=True, graph=False, dist_thresh=np.inf):

    data_points = tuple(_data_point(fid, size, labels, dist_lbl, graph, dist_thresh) for fid in sequences[sequence][::each])

    return data_points
