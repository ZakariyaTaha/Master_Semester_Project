import numpy as np
import sys
import os
import imageio
from skimage.morphology import skeletonize, skeletonize_3d, binary_dilation
from interruptions_score import interruptions_score
import cv2

output = "output"
folder_gt = "/cvlabdata2/home/kozinski/experimentsTorch/bbp_neurons/data_npy/lbl/test"
folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_retrace_v1/test_best_1"
#folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_v2_softerclassreweighting/test_best"

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
    
mkdir(output)

print("---name---", "---percentage connected---")    
ps = []
for i,(fgt, fpred) in enumerate(zip(filename_gts, filename_preds)):
    p = []
    for idx_maxprojection in range(3):
        label_s = load_gt(fgt, idx_max=idx_maxprojection, scale=2)
        pred_mask_s = load_pred(fpred, th=0.5, idx_max=idx_maxprojection, scale=2)

        p_connected, res = interruptions_score(label_s, pred_mask_s,
                                             radius_match=25, 
                                             th_similarity=0.35,
                                             intersections_only=False)

        _label_s = np.dstack([label_s.copy()]*3)*255
        _pred_mask_s = np.dstack([pred_mask_s.copy()]*3)*255
        for data in res:
            xs,ys = data["line_gt"][:,0], data["line_gt"][:,1]
            if data["connected"]:       
                _label_s[ys,xs] = np.array([0,255,0])
            else:
                _label_s[ys,xs] = np.array([255,0,0])
        for data in res:    
            if data["connected"]:    
                xs,ys = data["line_pred"][:,0], data["line_pred"][:,1]
                _pred_mask_s[ys,xs] = np.array([0,255,0])
        
        _comb = np.hstack([_label_s, _pred_mask_s[:,:10]*0+255, _pred_mask_s])
        _comb = cv2.putText(np.uint8(_comb), "{:0.2f}".format(p_connected), (30,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255,255,255])
        _root, _extension = os.path.splitext(os.path.basename(fgt))
        imageio.imsave(os.path.join(output, _root+"_{}".format(idx_maxprojection)+".jpg"), np.uint8(_comb))

        p.append(p_connected)
    print(os.path.basename(fgt), np.mean(p))
    ps.append(np.mean(p))
print("average", np.mean(ps))