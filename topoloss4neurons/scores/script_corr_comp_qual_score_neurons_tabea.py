import numpy as np
import sys
import os
import imageio
import pickle
from skimage.external import tifffile as tiff
from skimage.morphology import skeletonize_3d, binary_dilation
sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from correctness_completeness_quality_score import correctness_completeness_quality_3D, correctness, completeness, quality, f1

filename_volume_gt = "/cvlabsrc1/cvlab/datasets_leo/neurons_tabea/orig/lbl/test/acq_3_label_final.tif"
#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_baseline_grayscale2/prob_net_8000.tif"
filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_retracing/prob_net_8000.tif"
#filename_volume_pred = "/cvlabdata2/home/citraro/projects/delin/tabea_neurons/log_retracing2/prob_net_16000.tif"
output = os.path.dirname(filename_volume_pred)
filename_results = os.path.join(output, os.path.basename(filename_volume_pred).split('.')[0]+"_corr_comp_qual_results.txt")

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

    skeleton = skeletonize_3d(pred_mask>0)
    
    #print("I'm downscaling the predicted volume by a factor 4 to speed things up!!!!!")
    #skeleton = skeleton[::4,::4,::4]    
    
    return skeleton

def load_gt(filename):
    mask_gt = tiff.imread(filename)
    mask_gt = mask_gt.transpose((2,1,0))

    mask_gt = mask_gt.copy()
    # removing the some form the image
    mask_gt[1010:1060, 250:300, 162:192] = 0
    mask_gt[566:637, 190:552, 162:192] = 0
    
    #print("I'm downscaling the gt volume by a factor 4 to speed things up!!!!!")
    #mask_gt = mask_gt[::4,::4,::4]
    
    return mask_gt

print("="*20)
print(filename_volume_pred)
    
#mkdir(filename_volume_pred)

volume_gt_s = load_gt(filename_volume_gt)

file = open(filename_results, 'w')
file.write("slack threshold correctness completeness quality f1\n")
file.close()
file = open(filename_results, 'a')
   

for th in [0.3,0.4,0.5,0.6,0.7]:

    volume_pred_s = load_pred(filename_volume_pred, th)   

    for slack in [1, 3, 5]:
        _TP, _FN, _FP = correctness_completeness_quality_3D(volume_pred_s, volume_gt_s, slack=slack)
        TP, FN, FP = _TP.sum(), _FN.sum(), _FP.sum()

        corr = correctness(TP, FP, eps=1e-12)
        comp = completeness(TP, FN, eps=1e-12)
        qual = quality(TP, FP, FN, eps=1e-12)
        _f1 = f1(corr, comp, eps=1e-12)    

        file.write("{} {} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(slack, th, corr, comp, qual, _f1))
        print("slack={} th={} corr={:0.5f} comp={:0.5f} qual={:0.5f} f1={:0.5f}".format(slack, th, corr, comp, qual, _f1))
    
file.close()    
