
from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist

# ===================================================================================
image_height = -1
scales = {"orig":1}
base = "/cvlabdata2/home/zakariya/CarlsData/rendering/"
rendered_brains = ["rendered_AL175", "rendered_AL223", "rendered_AL230", "rendered_AL236", "rendered_AL242"]
tr = 90/100         # training portion
nb_cubes175 = 124            # nb of cubes of brain 175
nb_cubes223 = 82
nb_cubes230 = 219
nb_cubes236 = 124
nb_cubes242 = 94

path_brain={"175":os.path.join(base, "rendered_AL175"),
             "223":os.path.join(base, "rendered_AL223"),
             "230":os.path.join(base, "rendered_AL230"),
             "236":os.path.join(base, "rendered_AL236"), 
             "242":os.path.join(base, "rendered_AL242")}

path_cubes={"175":os.path.join(path_brain["175"], "cube_"),
             "223":os.path.join(path_brain["223"], "cube_"),
             "230":os.path.join(path_brain["230"], "cube_"),
             "236":os.path.join(path_brain["236"], "cube_"), 
             "242":os.path.join(path_brain["242"], "cube_")}

path_labels={"175":os.path.join(path_brain["175"], "label_"),
             "223":os.path.join(path_brain["223"], "label_"),
             "230":os.path.join(path_brain["230"], "label_"),
             "236":os.path.join(path_brain["236"], "label_"), 
             "242":os.path.join(path_brain["242"], "label_")}

path_distlabels={"175":os.path.join(path_brain["175"], "dist_"),
             "223":os.path.join(path_brain["223"], "dist_"),
             "230":os.path.join(path_brain["230"], "dist_"),
             "236":os.path.join(path_brain["236"], "dist_"), 
             "242":os.path.join(path_brain["242"], "dist_")}

rng_175 = np.random.permutation(nb_cubes175)
rng_223 = np.random.permutation(nb_cubes223)
rng_230 = np.random.permutation(nb_cubes230)
rng_236 = np.random.permutation(nb_cubes236)
rng_242 = np.random.permutation(nb_cubes242)
    
sequences = {"175": {
    "training": rng_175[:int(tr*nb_cubes175)],
    "testing": rng_175[int(tr*nb_cubes175):],
    " all" : rng_175},
             "230":{
    "training": rng_230[:int(tr*nb_cubes230)],
    "testing": rng_230[int(tr*nb_cubes230):],
    " all" : rng_230},
             "236":{
    "training": rng_236[:int(tr*nb_cubes236)],
    "testing": rng_236[int(tr*nb_cubes236):],
    " all" : rng_236}, 
             "242":{
    "training": rng_242[:int(tr*nb_cubes242)],
    "testing": rng_242[int(tr*nb_cubes242):],
    " all" : rng_242},
            "223":{
    "training": rng_223[:int(tr*nb_cubes223)],
    "testing": rng_223[int(tr*nb_cubes223):],
    " all" : rng_223}}

# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "dist_labels"])

def _data_point(fid, brain ,clip_value, size="train", labels="all", threshold=15):
    
    basename = str(fid)
    print("brain ", brain)
    filename = f'{path_cubes[brain]}{basename}.npy'
    image = np.load(filename).astype('float32') # as saved

    filename = f'{path_labels[brain]}{basename}.npy'
    label = np.load(filename).astype('uint8') # as saved

    dfilename = f'{path_distlabels[brain]}{basename}.npy'
    dist_labels = np.load(dfilename).astype('float32') # as saved
    dist_labels = np.clip(dist_labels, a_min=0, a_max=clip_value)

    return DataPoint(image, label, dist_labels)

def load_dataset(brains, clip_value, sequence='training', size="orig", labels="all", each=1, threshold=15):
    brains_str = list(map(str, brains))
    data_points = ()
    for brain in brains_str:
        data_points += tuple(_data_point(fid, brain, clip_value, size, labels, threshold) for fid in sequences[brain][sequence][::each])
   

    return data_points
