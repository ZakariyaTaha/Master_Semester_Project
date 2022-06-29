import numpy as np
import sys
import os
import imageio
import pickle
from skimage.external import tifffile as tiff
from skimage.morphology import skeletonize, binary_dilation
sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from correctness_completeness_quality_score import correctness_completeness_quality_2D, correctness, completeness, quality, f1

folder_gt = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/ours/test/labels/"
filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/isbi12_em/log_isbi12_baseline_v1/net_600/volume_prob.tif"
#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/isbi12_em/log_isbi12_retrace_rad10_alpha10_v1/net_600/volume_prob_sigma1.tif"
filename_results = os.path.join(filename_volume_pred.split('/')[-3]+"_"+filename_volume_pred.split('/')[-2]+"_corr_comp_qual_results.txt")

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

file = open(filename_results, 'w')
file.write("slack threshold correctness completeness quality f1\n")
file.close()
file = open(filename_results, 'a')
    
for slack in [1]:   
    for th in [0.6,0.7,0.8,0.9]:    
        ps = []
        for i,fgt in enumerate(filename_gts):

            slice_gt_s = load_gt(fgt)
            slice_pred_s = load_pred(filename_volume_pred, i, th)

            _TP, _FN, _FP = correctness_completeness_quality_2D(slice_pred_s, slice_gt_s, slack=slack)
            TP, FN, FP = _TP.sum(), _FN.sum(), _FP.sum()

            corr = correctness(TP, FP, eps=1e-12)
            comp = completeness(TP, FN, eps=1e-12)
            qual = quality(TP, FP, FN, eps=1e-12)
            _f1 = f1(corr, comp, eps=1e-12)    

            ps.append([corr, comp, qual, _f1])

        ps = np.array(ps)

        mean = np.mean(ps, axis=0)
        print("average:")
        print("slack", slack, "th", th, "correctness", mean[0], "completeness", mean[1], "quality", mean[2], "f1", mean[3])
        print("----------------------------------")

        file.write("{} {} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(slack, th, mean[0], mean[1], mean[2], mean[3]))
    
file.close()    
