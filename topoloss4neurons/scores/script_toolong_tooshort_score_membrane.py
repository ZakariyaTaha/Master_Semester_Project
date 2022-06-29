import numpy as np
import sys
import os
import imageio
import pickle
from skimage.external import tifffile as tiff
from skimage.morphology import skeletonize, binary_dilation
sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from toolong_tooshort_score import find_connectivity, create_graph, extract_gt_paths, toolong_tooshort_score

folder_gt = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/ours/test/labels/"
filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/isbi12_em/log_isbi12_baseline_v1/net_600/volume_prob.tif"

#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/isbi12_em/log_isbi12_retrace_rad10_alpha10_v1/net_600/volume_prob_sigma1.tif"
filename_results = os.path.join(filename_volume_pred.split('/')[-3]+"_"+filename_volume_pred.split('/')[-2]+"_2long_2short_results.txt")
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

def load_pred(filename, i, th=0.5):
    volume_pred = tiff.imread(filename)<th
    slice_pred_s = skeletonize(volume_pred[i].copy())>0
    return slice_pred_s

def load_gt(filename):
    slice_gt = tiff.imread(filename)>0
    slice_gt_s = skeletonize(slice_gt.copy())>0
    return slice_gt_s

print("="*20)
print(filename_volume_pred)

filename_gts = find_files(folder_gt, "*")
print("filename_gts:")
for f in filename_gts:
    print(f)
'''
filename_preds = find_files(folder_pred, "*")
print("filename_preds:")
for f in filename_preds:
    print(f)
'''    
#mkdir(output)                        
                                    
if load_saved_paths:
    pathss = pickle_read("/home/citraro/projects/topoloss4neurons/topo_score/paths_2long_2short_membrane2.pickle")
else:
    pathss = []
    for i,fgt in enumerate(filename_gts):
        volume_gt_s = load_gt(fgt)
        graph_gt = create_graph(volume_gt_s)    
    
        paths = extract_gt_paths(graph_gt, 
                                 N=200, 
                                 min_path_length=10)
        print("n_paths_gt", len(paths), "mean_length", np.mean([len(x["shortest_path_gt"]) for x in paths]))        
        pathss.append(paths)       
    
    pickle_write("/home/citraro/projects/topoloss4neurons/topo_score/paths_2long_2short_membrane2.pickle",pathss)
    
print("-->File with results: ", filename_results)    
    
file = open(filename_results, "w")
file.write("threshold correct tooshort toolong infeasible\n")
file.close()
file = open(filename_results, "a")
    
for th in [0.7]:    
    ps = []
    for i,fgt in enumerate(filename_gts):

        volume_pred_s = load_pred(filename_volume_pred, i, th)

        graph_pred = create_graph(volume_pred_s)

        tot,correct,tooshort,toolong,infeasible,res = toolong_tooshort_score(pathss[i], 
                                                                             graph_pred, 
                                                                             radius_match=15, 
                                                                             length_deviation=0.1)

        ps.append([correct,tooshort,toolong,infeasible])

        #print("th", th, "name",os.path.basename(fgt), "correct", correct, \
        #      "tooshort", tooshort, "toolong", toolong, "infeasible", infeasible)

    ps = np.array(ps)

    mean = np.mean(ps, axis=0)
    print("average:")
    print("th", th, "correct", mean[0], "tooshort", mean[1], "toolong", mean[2], "infeasible", mean[3])
    print("----------------------------------")
          
    file.write("{} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(th, mean[0], mean[1], mean[2], mean[3]))
    
file.close()    
