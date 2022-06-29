import numpy as np
import networkx as nx
import apls_mod_3d

__all__ = ["dice_score", "confusion_matrix", 
           "precision_recall", "f1", "accuracy", "apls"]

def dice_score(pred, mask, eps=1e-6):
    """ Computes Dice score
    
    Parameters
    ----------
    pred : numpy.bool_
        predicted mask
    truth : numpy.bool_
        true mask    
    """  

    union = pred.sum() + mask.sum()
    intersection = (pred*mask).sum()

    return 2*intersection/(union + eps)

def confusion_matrix(pred, truth):
    """ Computes confusione matrix
    
    Parameters
    ----------
    pred : numpy.bool_
        predicted mask
    truth : numpy.bool_
        true mask    
        
    Return
    ------
    true-positives, true-negatives
    false-positives, false-negatives
    """
    
    mask_P = (truth==True)[0]
    mask_N = (truth==False)[0]
    
    TP = np.sum(pred[mask_P]==True)
    TN = np.sum(pred[mask_N]==False)
    FN = np.sum(pred[mask_P]==False)
    FP = np.sum(pred[mask_N]==True) 
    
    return TP, TN, FP, FN

def _precision(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps)

def _recall(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps)

def _f1(precision, recall, eps=1e-12):
    return 2.0/(1.0/(precision+eps) + 1.0/(recall+eps))
    
def _accuracy(TP, TN, FP, FN, eps=1e-12):
    return (TP + TN)/(TP + TN + FP + FN + eps)

def precision_recall(pred, truth, eps=1e-12):
    TP, TN, FP, FN = confusion_matrix(pred, truth)
    return _precision(TP, FP, eps), _recall(TP, FN, eps)

def f1(pred, truth, eps=1e-12):
    p,r = precision_recall(pred, truth, eps)
    return _f1(p, r, eps)
    
def accuracy(pred, truth, eps=1e-12):
    TP, TN, FP, FN = confusion_matrix(pred, truth)
    return _accuracy(TP, TN, FP, FN, eps)

def apls(graph_pred, graph_gt):
    conct_threshold=50
    weight='length'
    max_snap_dist=25          
    dist_close_node=25
    max_nodes=2000
    G_gt_cp2, G_p_cp2, G_gt_cp_prime2, G_p_cp_prime2, \
    control_points_gt2, control_points_prop2, \
    all_pairs_lengths_gt_native2, all_pairs_lengths_prop_native2, \
    all_pairs_lengths_gt_prime2, all_pairs_lengths_prop_prime2  \
    = apls_mod_3d.make_graphs_yuge(graph_gt, graph_pred,
                          weight=weight,
                          max_snap_dist=max_snap_dist,
                          dist_close_node=dist_close_node,
                          allow_renaming=True,
                          verbose=False,
                          max_nodes=max_nodes,
                          select_intersections=True)
    print("Make graph done")
    C2, C_gt_onto_prop2, C_prop_onto_gt2 = apls_mod_3d.compute_apls_metric(
            all_pairs_lengths_gt_native2, all_pairs_lengths_prop_native2,
            all_pairs_lengths_gt_prime2, all_pairs_lengths_prop_prime2,
            control_points_gt2, control_points_prop2,
            min_path_length=40,
            verbose=False, super_verbose=False, res_dir=".")
    print(C2, C_gt_onto_prop2, C_prop_onto_gt2)
    return C2
