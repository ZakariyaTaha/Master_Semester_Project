from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def reweightErrors(pred, gt, exponent, iters):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1

    pos=gt
    gt_dil = binary_dilation(gt, iterations=iters)
    neg=np.logical_not(gt_dil)
    
    w=pos+pred*(neg-pos)
    w=w**exponent
    
    return w

def reweightErrors_noDilation(pred, gt, exponent):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1

    pos=gt
    neg=np.logical_not(gt)
    
    w=pos+pred*(neg-pos)
    w=w**exponent
    
    return w


def reweightErrors_v2(pred, gt, exponent, iters):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1

    # positive area - on ground truth neurons
    pos=gt
    gt_dil = binary_dilation(pos, iterations=iters)
    # negative area - "iters" pixels away from positive
    neg=np.logical_not(gt_dil)
    # dont know what - in between
    ignored=np.logical_not(np.logical_or(neg,pos))
    
    # reweight positive and negatives, keep weight=1 for the third area
    w_neg=(neg*pred)**exponent
    w_neg=w_neg*neg.sum()/w_neg.sum()
    w_pos=(pos*(1-pred))**exponent
    w_pos=w_pos*pos.sum()/w_pos.sum()
    return w_neg+w_pos+ignored

def reweightErrors_v2_onlyPos(pred, gt, exponent, iters):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1

    # positive area - on ground truth neurons
    pos=gt
    neg=1-pos
    
    # reweight positive and negatives, keep weight=1 for the third area
    w_pos=(pos*(1-pred))**exponent
    w_pos=w_pos*pos.sum()/w_pos.sum()
    return w_pos+neg

def reweightErrors_v3_Doruk(pred, gt, iters):
    # pred is within the range of [0,1]
    # the class for foreground is gt==1
    # positive - 
    wfalseneg=2
    wfalseneg_certain=0.5
    wtruepos=1
    # negative
    wfalsepos=2
    wfalsepos_certain=0.5
    wtrueneg=1
    wdontknow=0.5

    highpred=pred>=0.5
    lowpred= pred<0.5
    toohigh=pred>0.95
    toolow =pred<0.05
    # 0.5<not2high<0.95
    not2high=np.logical_and(highpred,np.logical_not(toohigh))
    # 0.5>not2low >0.05
    not2low =np.logical_and(lowpred, np.logical_not(toolow))
    # positive area - on ground truth neurons
    pos=gt
    gt_dil = binary_dilation(pos, iterations=iters)
    # negative area - "iters" pixels away from positive
    neg=np.logical_not(gt_dil)
    # area between pos and neg
    dontknow=np.logical_not(np.logical_or(neg,pos))

    falsepos=np.logical_and(neg,not2high)
    falsepos_certain=np.logical_and(neg,toohigh)
    trueneg=np.logical_and(neg,lowpred)

    falseneg=np.logical_and(pos,not2low)
    falseneg_certain=np.logical_and(pos,toolow)
    truepos=np.logical_and(pos,highpred)
    assert(np.all(falseneg_certain+falseneg+truepos \
                 +falsepos_certain+falsepos+trueneg + dontknow))

    wpos=wfalseneg_certain*falseneg_certain+wfalseneg*falseneg+truepos*wtruepos
    wpos=wpos*pos.sum()/wpos.sum() # do not change the balance of classes!
    wneg=wfalsepos_certain*falsepos_certain+wfalsepos*falsepos+trueneg*wtrueneg+dontknow*wdontknow
    wneg=wneg*neg.sum()/wneg.sum() # do not change the balance of classes!

    return wneg+wpos

