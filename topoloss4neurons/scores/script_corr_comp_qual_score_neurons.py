import numpy as np
import sys
import os
import imageio
import pickle
from skimage.external import tifffile as tiff
from skimage.morphology import skeletonize_3d, binary_dilation
sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from correctness_completeness_quality_score import correctness_completeness_quality_3D, correctness, completeness, quality, f1

folder_gt = "/cvlabdata2/home/kozinski/experimentsTorch/bbp_neurons/data_npy/lbl/test"
#folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_retrace_v1/test_best_1"
folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_v2_softerclassreweighting/test_best"
#folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_retrace_v5_reweighting_only/last_net_test_set_1"
filename_results = os.path.join(folder_pred.split('/')[-2]+"_"+folder_pred.split('/')[-1]+"_corr_comp_qual_results.txt")

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
    volume_pred = np.load(fpred)>th
    volume_pred_s = skeletonize_3d(volume_pred.copy())>128
    return volume_pred_s

def load_gt(filename):
    volume_gt = np.load(fgt)
    volume_gt = volume_gt-1
    volume_gt[volume_gt==255] = 0
    volume_gt = volume_gt>0

    volume_gt_s = skeletonize_3d(volume_gt.copy())>128
    return volume_gt_s

print("="*20)
print(folder_pred)

filename_gts = find_files(folder_gt, "*")
print("filename_gts:")
for f in filename_gts:
    print(f)

filename_preds = find_files(folder_pred, "*")
print("filename_preds:")
for f in filename_preds:
    print(f)    

file = open(filename_results, 'w')
file.write("slack threshold correctness completeness quality f1\n")
file.close()
file = open(filename_results, 'a')
   
for slack in [1, 3, 5]:
    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

        ps = []
        for i,(fgt, fpred) in enumerate(zip(filename_gts, filename_preds)): 

            volume_gt_s = load_gt(fgt)    
            volume_pred_s = load_pred(fpred, th)   

            _TP, _FN, _FP = correctness_completeness_quality_3D(volume_pred_s, volume_gt_s, slack=slack)
            TP, FN, FP = _TP.sum(), _FN.sum(), _FP.sum()

            corr = correctness(TP, FP, eps=1e-12)
            comp = completeness(TP, FN, eps=1e-12)
            qual = quality(TP, FP, FN, eps=1e-12)
            _f1 = f1(corr, comp, eps=1e-12)    
            
            ps.append([corr, comp, qual, _f1])
        mean = np.array(ps).mean(0)

        file.write("{} {} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(slack, th, mean[0], mean[1], mean[2], mean[3]))
        print("slack={} th={} corr={:0.5f} comp={:0.5f} qual={:0.5f} f1={:0.5f}".format(slack, th, mean[0], mean[1], mean[2], mean[3]))
    
file.close()    