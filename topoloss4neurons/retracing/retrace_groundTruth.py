import torch
import numpy as np
from networkTraining.forwardOnBigImages import processChunk
from detectTopoErrors_v1 import detectTopoErrors
import os
from skimage.external import tifffile as tiff
from retracing_routines import resamplePaths, resamplePaths2D
from detectReweightedErrors import reweightErrors, reweightErrors_noDilation, \
  reweightErrors_v2, reweightErrors_v2_onlyPos, reweightErrors_v3_Doruk

def process_output(o):
    e=np.exp(o[0,1])
    prob=e/(e+1)
    return prob

class retracer:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    
  def test(self,net):
    
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        
        self.dataset.lbl[i]=lbl
    net.train()       
        
class retracer_reweighter:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,exponent=2,dilation=None):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.exponent=exponent
    self.dilation = dilation if dilation!=None else self.rad

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors(prob,lbl,self.exponent,self.dilation)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)

    net.train()
        
class reweighter:
  def __init__(self, train_dataset,logdir,exponent=2):
    self.dataset=train_dataset
    self.logdir=logdir
    self.exponent=exponent

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        w=reweightErrors_noDilation(prob,lbl,self.exponent)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)

    net.train()
        
class retracer_reweighter_noDilation:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,exponent=2):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.exponent=exponent

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors_noDilation(prob,lbl,self.exponent)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)

    net.train()
        
class retracer_reweighter_v2:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,exponent=2,dilation=None):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.exponent=exponent
    self.dilation = dilation if dilation!=None else self.rad

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors_v2(prob,lbl,self.exponent,self.dilation)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)

    net.train()
        
class retracer_reweighter_v2_onlyPos:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,exponent=2,dilation=None):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.exponent=exponent
    self.dilation = dilation if dilation!=None else self.rad

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        img=self.dataset.img[i]
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(104,104,104),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors_v2_onlyPos(prob,lbl,self.exponent,self.dilation)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)

    net.train()
        
class retracer_reweighter_v3_Doruk:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,dilation=None):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.dilation = dilation if dilation!=None else self.rad
    self.j = 0

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        
        print("Doing retracing of image {}".format(i))
        
        img=self.dataset.img[i].astype(np.float32)/255
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        inp=img.reshape(1,1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,(128,128,128),(22,22,22),2,net,outChannels=2)
        prob=process_output(oup)
        np.save(os.path.join(self.logdir,"prob_"+str(i)+"_iter"+str(self.j)+".npy"),prob) 
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        lbl=resamplePaths(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors_v3_Doruk(prob,lbl,self.dilation)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)
        
        #if i==0:
        np.save(os.path.join(self.logdir,"lbl_"+str(i)+"_iter"+str(self.j)+".npy"),lbl) 
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_iter"+str(self.j)+".npy"),w)          

    net.train()
    self.j += 1    
        
class retracer_reweighter_v3_Doruk2D:
  def __init__(self, train_dataset,logdir,rad=5,alpha=4,dilation=None,cropSz=(256,256),inteVal=(22,22)):
    self.dataset=train_dataset
    self.logdir=logdir
    self.rad=rad
    self.alpha=alpha
    self.dilation = dilation if dilation!=None else self.rad
    
    self.cropSz = cropSz
    self.inteVal = inteVal
    self.j = 0

  def test(self,net):
    net.eval()
    with torch.no_grad():
      for i in range(len(self.dataset.img)):
        
        print("Doing retracing of image {}".format(i))
        
        img=self.dataset.img[i].astype(np.float32)/255
        lbl=self.dataset.lbl[i]
        edges=self.dataset.edges[i]
        node_coords=self.dataset.node_coords[i]
        
        inp=img.reshape(1,img.shape[-3],img.shape[-2],img.shape[-1])
        oup=processChunk(inp,self.cropSz,self.inteVal,2,net,outChannels=2)
        
        prob=process_output(oup)
        
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_prev.npy"),lbl)
        lbl.fill(0)
        
        lbl=resamplePaths2D(lbl,prob,edges,node_coords,self.rad,self.alpha)
        np.save(os.path.join(self.logdir,"lbl"+str(i)+"_last.npy"),lbl)
        self.dataset.lbl[i]=lbl

        w=reweightErrors_v3_Doruk(prob,lbl,self.dilation)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_old.npy"),
                self.dataset.weight[i])
        self.dataset.weight[i]=w.astype(np.float32)
        np.save(os.path.join(self.logdir,"weight_"+str(i)+".npy"),w)
        
        #if i==0:
        np.save(os.path.join(self.logdir,"lbl_"+str(i)+"_iter"+str(self.j)+".npy"),lbl) 
        np.save(os.path.join(self.logdir,"weight_"+str(i)+"_iter"+str(self.j)+".npy"),w)

    net.train()
    
    self.j += 1
        
