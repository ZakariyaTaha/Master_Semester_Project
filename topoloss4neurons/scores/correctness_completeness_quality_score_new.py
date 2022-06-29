import numpy as np
from scipy import ndimage

__all__ = ["correctness_completeness_quality_new"]

def correctness_new(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps) # precision

def completeness_new(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps) # recall
    
def quality_new(TP, FP, FN, eps=1e-12):
    return TP/(TP + FP + FN + eps)
    
def f1_new(correctness, completeness, eps=1e-12):
    return 2.0/(1.0/(correctness+eps) + 1.0/(completeness+eps))

def relaxed_confusion_matrix_new(pred_s, gt_s, gt_d, slack=3):
    
    distances_gt = gt_d#ndimage.distance_transform_edt((np.logical_not(gt_s)))
    distances_pred = np.float32(ndimage.distance_transform_edt((np.logical_not(pred_s))))
    
    true_pos_area = distances_gt<=slack
    false_pos_area = distances_gt>slack   
    false_neg_area = distances_pred>slack
    
    true_positives = np.logical_and(true_pos_area, pred_s).sum()
    false_positives = np.logical_and(false_pos_area, pred_s).sum()
    false_negatives = np.logical_and(false_neg_area, gt_s).sum()

    return true_positives, false_negatives, false_positives

def correctness_completeness_quality_new(pred_s, gt_s, gt_d, slack=3, eps=1e-12):
    
    TP, FN, FP = relaxed_confusion_matrix_new(pred_s, gt_s, gt_d, slack)

    return correctness_new(TP, FP, eps), completeness_new(TP, FN, eps), quality_new(TP, FP, FN, eps)

