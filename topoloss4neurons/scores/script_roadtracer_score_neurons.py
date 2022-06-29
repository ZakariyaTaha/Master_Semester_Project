import numpy as np
import sys
import os
import imageio
from skimage.morphology import skeletonize, skeletonize_3d, binary_dilation
from roadtracer_score import roadtracer_score

folder_gt = "/cvlabdata2/home/kozinski/experimentsTorch/bbp_neurons/data_npy/lbl/test"
folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_retrace_v1/test_best_1"
#folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_v2_softerclassreweighting/test_best"

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

def sigmoid(x):
    e = np.exp(x)
    return e/(e + 1)

def load_pred(filename, th=0.5, idx_max=None, scale=8):
    volume = np.load(filename)
    #print(volume.shape, volume.min(), volume.max(), volume.dtype)

    volume = volume>0.5 # do I need to sigmoid it?
    #volume = binary_dilation(volume)
    #volume = scipy.ndimage.zoom(volume, scale, order=0)
    #volume = volume.repeat(scale,axis=0).repeat(scale,axis=1).repeat(scale,axis=2)
    volume = skeletonize_3d(volume>th)
    if idx_max is not None:
        # to remove some noise
        max_proj = volume.max(idx_max)
        #max_proj = rescale(max_proj, scale=scale, order=2, mode='constant')
        max_proj = max_proj.repeat(scale,axis=0).repeat(scale,axis=1)
        max_proj = skeletonize(binary_dilation(binary_dilation(skeletonize(max_proj>0))))
        #max_proj = rescale(max_proj, scale=scale, order=2, mode='constant')
        return skeletonize(max_proj>0.5)
    return volume

def load_gt(filename, idx_max=None, scale=8):
    volume = np.load(filename)
    volume = volume-1
    volume[volume==255] = 0
    #print(volume.shape, volume.min(), volume.max(), volume.dtype)

    volume = volume>0.5 # do I need a sigmoid?
    #volume = binary_dilation(volume)
    #volume = scipy.ndimage.zoom(volume, scale, order=0)
    #volume = volume.repeat(scale,axis=0).repeat(scale,axis=1).repeat(scale,axis=2)
    volume = skeletonize_3d(volume)
    if idx_max is not None:
        # to remove some noise
        max_proj = volume.max(idx_max)
        #max_proj = rescale(max_proj, scale=scale, order=2, mode='constant')
        max_proj = max_proj.repeat(scale,axis=0).repeat(scale,axis=1)
        max_proj = skeletonize(binary_dilation(binary_dilation(skeletonize(max_proj>0))))
        #max_proj = rescale(max_proj, scale=scale, order=2, mode='constant')
        return skeletonize(max_proj>0.5)   
    return volume

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

print("---name---", "---f_v_correct---", "---f_u_error---")    
fca = []
fea = []
for i,(fgt, fpred) in enumerate(zip(filename_gts, filename_preds)):
    fc = []
    fe = []
    for idx_maxprojection in range(3):
        label_s = load_gt(fgt, idx_max=idx_maxprojection, scale=2)
        pred_mask_s = load_pred(fpred, th=0.5, idx_max=idx_maxprojection, scale=2)

        f_v_correct, f_u_error, debugs = roadtracer_score(label_s, pred_mask_s, 
                                                          radius_match=30, 
                                                          radius_directions=6, 
                                                          clustering_d=6)
        
        fc.append(f_v_correct)
        fe.append(f_u_error)
    print(os.path.basename(fgt), np.mean(fc), np.mean(fe))
    fca.append(np.mean(fc))
    fea.append(np.mean(fe))
print("average", np.mean(fca), np.mean(fea))