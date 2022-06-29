import numpy as np
import sys
import os
import imageio
import pickle
from skimage.external import tifffile as tiff
from skimage.morphology import skeletonize_3d, binary_dilation
sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from toolong_tooshort_score import find_connectivity_3d, create_graph_3d, extract_gt_paths, toolong_tooshort_score

filename_volume_gt = "/cvlabsrc1/cvlab/datasets_leo/neurons_tabea/orig/lbl/test/acq_3_label_final.tif"
#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_baseline_grayscale2/prob_net_8000.tif"
filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_retracing/prob_net_8000.tif"
#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_retracing2/prob_net_16000.tif"
output = os.path.dirname(filename_volume_pred)
filename_results = os.path.join(output, os.path.basename(filename_volume_pred).split('.')[0]+"_2long_2short_results.txt")
load_saved_paths = True

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    import re
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None): 
    import os, glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data
        
def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def sigmoid(x):
    e = np.exp(x)
    return e/(e + 1)

def load_pred(filename, th=0.5):

    prob = tiff.imread(filename)    
    pred_mask = prob>th

    pred_mask = pred_mask.copy()
    # removing the some form the image
    pred_mask[1010:1060, 250:300, 162:192] = 0
    pred_mask[566:637, 190:552, 162:192] = 0

    skeleton = skeletonize_3d(pred_mask)
    return skeleton

def load_gt(filename):
    mask_gt = tiff.imread(filename)
    mask_gt = mask_gt.transpose((2,1,0))

    mask_gt = mask_gt.copy()
    # removing the some form the image
    mask_gt[1010:1060, 250:300, 162:192] = 0
    mask_gt[566:637, 190:552, 162:192] = 0
    return mask_gt

print("="*20)
print(filename_volume_pred)
    
#mkdir(filename_volume_pred)

volume_gt_s = load_gt(filename_volume_gt)
graph_gt = create_graph_3d(volume_gt_s)

if load_saved_paths:
    paths = pickle_read("/home/citraro/projects/topoloss4neurons/topo_score/paths_2long_2short_tabeas_neurons.pickle")
else:
    paths = extract_gt_paths(graph_gt, 
                             N=2000, 
                             min_path_length=50)
    print("n_paths_gt", len(paths), "mean_length", np.mean([len(x["shortest_path_gt"]) for x in paths]))
    pickle_write("/cvlabdata2/home/oner/Snakes/tabe_paths/paths_2long_2short_tabea.pickle",pathss)
    
file = open(filename_results, 'w')
file.write("threshold correct tooshort toolong infeasible\n")
file.close()
file = open(filename_results, 'a')
   
#for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
for th in [0.3,0.4,0.5,0.6,0.7]:

    volume_pred_s = load_pred(filename_volume_pred, th)   
    
    graph_pred = create_graph_3d(volume_pred_s)  

    tot,correct,tooshort,toolong,infeasible,res = toolong_tooshort_score(paths, 
                                                                         graph_pred, 
                                                                         radius_match=30, 
                                                                         length_deviation=0.15)
    
    
    file.write("{} {:0.5f} {:0.5f} {:0.5f} {:0.5f}".format(th, correct, tooshort, toolong, infeasible))
    print("th={} correct={:0.5f} tooshort={:0.5f} toolong={:0.5f} infeaseable={:0.5f}\n".format(th, correct, tooshort, toolong, infeasible))
    
file.close()    
