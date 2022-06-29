from skimage.morphology import skeletonize, skeletonize_3d, binary_dilation
import numpy as np

def _detect_topo_errors(pred_s, gt_s, slack=3):
    
    distances_gt = ndimage.distance_transform_edt((np.logical_not(gt_s)))
    distances_pred = ndimage.distance_transform_edt((np.logical_not(pred_s)))
    
    true_pos_area = distances_gt<=slack
    false_pos_area = distances_gt>slack   
    false_neg_area = distances_pred>slack
    
    true_positives = np.logical_and(true_pos_area, pred_s)
    false_positives = np.logical_and(false_pos_area, pred_s)
    false_negatives = np.logical_and(false_neg_area, gt_s)

    return true_positives, false_negatives, false_positives

def detect_topo_errors_2d(pred, gt, threshold=0.5, slack=3):
    pred_bin = pred > threshold

    pred_s = skeletonize(pred_bin)
    gt_s = skeletonize(gt==1)    
    
    return _detectTopoErrors(pred_s, gt_s, slack)

def detect_topo_errors_3d(pred, gt, threshold=0.5, slack=3):
    pred_bin = pred > threshold

    pred_s = skeletonize_3d(pred_bin)
    gt_s = skeletonize_3d(gt==1)    
    
    return _detectTopoErrors(pred_s, gt_s, slack)
'''
def detectTopoErrors(pred, gt):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1
    # Thresholding and skeletonization
    threshold=0.5
    pred_bin = pred > threshold

    pred_s = skeletonize_3d(pred_bin)
    gt_s = skeletonize_3d(gt==1)

    gt_dil = binary_dilation(gt_s, iterations=3)
    
    # Find the topologically acceptable range for each gt pixel
    indices = np.zeros(((np.ndim(gt_s),) + gt_s.shape), dtype=np.int32)
    ndimage.distance_transform_edt((np.logical_not(gt_s)), return_indices=True, indices=indices)
    
    # find holes and false positives
    gt_predicted=np.zeros_like(gt_s,dtype=np.bool8) # gt pixels that were predicted
    inds=np.nonzero(gt_dil)
    for x,y,z in zip(*inds):
        gt_predicted[indices[0,x,y,z],indices[1,x,y,z],indices[2,x,y,z]]=\
          gt_predicted[indices[0,x,y,z],indices[1,x,y,z],indices[2,x,y,z]] or pred_s[x,y,z]
                                     
    gt_notPredicted=np.logical_and(gt_s,np.logical_not(gt_predicted)) # gt pixels without corresponding predictions
                                   
    false_positive_predictions=np.logical_and(pred_s,np.logical_not(gt_dil))
    
    return gt_predicted, gt_notPredicted, false_positive_predictions
'''    