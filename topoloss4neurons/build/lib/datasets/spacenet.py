from collections import namedtuple
import os
import numpy as np
import imageio
import time
from skimage.morphology import skeletonize
import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
from .. import utils

# ===================================================================================
### Images converted by orientation & segmentation guys
base = "/cvlabdata1/home/oner/Spacenet/full/images/"
### Images converted by Spacenet & APLS guys
# base = "/cvlabdata1/home/oner/Spacenet/full_apls/images/"

base_gt = "/cvlabdata1/home/oner/Spacenet/full/labels/"
path_images={"train":base,
             "test" :base}
path_test_images={"orig":os.path.join(base, 'imagery_test'),
             "half" :os.path.join(base, 'imagery_test')}
path_labels={"orig":os.path.join(base, 'masks_thick'),
             "half" :os.path.join(base, 'masks_thick')}
path_labels_thin={"orig":os.path.join(base, 'masks'),
                  "half" :os.path.join(base, 'masks')}
path_labels_dist={"train":os.path.join(base, 'dist_lbl'),
                  "test" :os.path.join(base, 'dist_lbl')}

with open('/cvlabdata2/home/oner/MALIS/road_connectivity/data/spacenet/train.txt') as f:
	train = f.read()

with open('/cvlabdata2/home/oner/MALIS/road_connectivity/data/spacenet/val.txt') as f:
	val = f.read()

train_names = train.split("\n")
test_names = val.split("\n")

sequences = {"training": train_names,
             "testing":  test_names,
             "trial": train_names[:160],
             "all":      train_names}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "basename", "filename"])

def _data_point(fid, size="train", labels="all", dist_lbl=True, graph=False, dist_thresh=20):
    s = time.time()
    basename = fid
    filename = fid
    name = "SN3_roads_train_AOI_" + fid.split("_")[2] + "_" + fid.split("_")[3] + "_PS-RGB_" + fid.split("_")[4] + ".png"
    image = imageio.imread(os.path.join(base, name))

    label = None
    lbl_name = "AOI_" + fid.split("_")[2] + "_" + fid.split("_")[3] + "_PS-RGB_" + fid.split("_")[4] + ".png"
    label_dist = dist(~skeletonize(imageio.imread(os.path.join(base_gt, lbl_name))//150))
    label_dist[label_dist > dist_thresh] = dist_thresh
    print(time.time() - s)
    return DataPoint(image, label, label_dist, basename, filename)

def load_dataset(sequence='training', size="train", labels="all", each=1, dist_lbl=True, graph=False, dist_thresh=np.inf):
    ss = time.time()
    data_points = tuple(_data_point(fid, size, labels, dist_lbl, graph, dist_thresh) for fid in sequences[sequence][::each])
    print("Total time to load: " + str(time.time() - ss) + " seconds")
    return data_points
