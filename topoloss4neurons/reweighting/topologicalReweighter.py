import torch
import numpy as np
from networkTraining.forwardOnBigImages import processChunk
from networkTraining.utils import sigmoid, torch_no_grad
from detectTopoErrors_v1 import detectTopoErrors
import os

class topoReweighter:
    def __init__(self, train_dataset, logdir, factor=10.0, factor_falsePos=0.0):
        
        self.dataset = train_dataset
        self.logdir = logdir
        self.factor = factor
        self.factor_falsePos = factor_falsePos        
        
    def test(self, net):
        net.eval()
        
        with torch_no_grad:
            for i in range(len(self.dataset.img)):
                img = self.dataset.img[i]
                lbl = self.dataset.lbl[i]
                inp = img.reshape(1, 1, img.shape[-3], img.shape[-2], img.shape[-1]) # what is that!
                
                oup = processChunk(inp, (104,104,104), (22,22,22), 2, net, outChannels=2)
                prob = sigmoid(oup[0,1])
                
                detected, missed, falsepos = detectTopoErrors(prob, lbl.copy())
                missed, falsepos = np.float32(missed), np.float32(falsepos)
                
                w = np.ones_like(lbl) + self.factor*missed + self.factor_falsePos*falsepos
                w = np.float32(w/w.sum())
                
                filename_old = os.path.join(self.logdir, "weight_{}_old.npy".format(i))
                np.save(filename_old, self.dataset.weight[i])
                
                self.dataset.weight[i] = w.astype(np.float32)
                
                filename_new = os.path.join(self.logdir,"weight_{}.npy".format(i))
                np.save(filename_new, w)            
 
        net.train()
        
