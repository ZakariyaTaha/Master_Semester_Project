import os
import sys
import time
import torch
import logging
import argparse
import itertools
import numpy as np
from copy import deepcopy
from shutil import copyfile
from collections import namedtuple
from skimage.external import tifffile
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import skeletonize

import networkTraining_mod as nt
import topoloss4neurons_mod as tl
from topoloss4neurons_mod.networks import UNet
from topoloss4neurons_mod import load_dataset
from topoloss4neurons_mod.retracing.retracing_routines import resamplePaths2D
from topoloss4neurons_mod.reweighting.detectReweightedErrors import reweightErrors_v3_Doruk

logger = logging.getLogger(__name__)

ExtDataPoint = namedtuple("ExtDataPoint", ['image', 'label', 'label_ignore', 'weights', 'node_coords', 'edges'])

IGNORE = 255
'''
def compute_class_weights(labels, num_classes):
    bc = nt.BinCounter(num_classes)
    for labels_i in labels:
        bc.update(labels_i)
        
    class_weights = 1.0 / (num_classes * bc.frequencies())
    class_weights[np.isinf(class_weights)] = np.max(class_weights[~np.isinf(class_weights)])
    return np.float32(class_weights)
'''
def compute_class_weights(labels, num_classes):
    bc = nt.BinCounter(num_classes)
    for labels_i in labels:
        bc.update(labels_i)
        
    freq = bc.frequencies() + 1.02
    freq = np.log(freq)
    w = 1/freq
    w = w/w.sum()
    return w 

class Retracing(object):    
    
    def __init__(self, crop_size, margin_size, dataset, out_channels, output_path,
                 rad=10, alpha=100, dilation=5):    
        
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataset = dataset 
        self.out_channels = out_channels
        self.output_path = output_path
        
        self.rad = rad
        self.alpha = alpha
        self.dilation = dilation if dilation!=None else rad
    
    def __call__(self, iteration, network):
        
        logger.info("Retracing...")
        start_time = time.time()

        network.train(False)        
        with nt.torch_no_grad:
            for i,dp in enumerate(self.dataset):
                
                logger.info("\t [{:0.2f}s] retracing sample {}...".format(time.time()-start_time, i))
                
                image_i = np.transpose(np.float32(dp.image[None])/255.0, (0, 3, 1, 2))                
                image_i  = nt.to_torch(image_i, volatile=True).cuda()                           

                out_shape = (image_i.shape[0],self.out_channels,*image_i.shape[2:])
                pred_i = nt.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred_i = nt.process_in_chuncks(image_i, pred_i, 
                                               lambda chunk: network(chunk), 
                                               self.crop_size, self.margin_size) 
                
                prob_i = nt.softmax(nt.from_torch(pred_i)[0])[1]
                
                new_label_i = np.zeros_like(dp.label, np.uint8)                
                new_label_i = resamplePaths2D(new_label_i, prob_i, 
                                              dp.edges, dp.node_coords, 
                                              self.rad, self.alpha)

                weights_i = reweightErrors_v3_Doruk(prob_i, new_label_i, self.dilation)
                
                self.dataset[i] = ExtDataPoint(dp.image, new_label_i, new_label_i, weights_i, 
                                               dp.node_coords, dp.edges)
                
                # to save stuff
                if True:
                    output = os.path.join(self.output_path, "output_retracing")
                    nt.mkdir(output)
                    tifffile.imsave(os.path.join(output, "label_{:06d}_iter{:06d}.tiff".format(i,iteration)), new_label_i)
                    tifffile.imsave(os.path.join(output, "weights_{:06d}_iter{:06d}.tiff".format(i,iteration)), weights_i)

        network.train(True)

class TrainingStep(object):
    
    def __init__(self, batch_size, crop_size, dataset, probas,
                 retrace_every, retracing):
        
        self.batch_size = batch_size  
        self.crop_size = crop_size
        self.dataset = dataset
        self.probas = probas
        self.retrace_every = retrace_every
        self.retracing = retracing
        
    def __call__(self, iterations, network, optimizer, lr_scheduler, loss_function):

        replace = True if len(self.dataset)<self.batch_size else False
        minibatch = [self.dataset[i] for i in np.random.choice(np.arange(len(self.dataset)), 
                                                               self.batch_size, 
                                                               replace=replace,
                                                               p=self.probas)]

        images = [dp.image for dp in minibatch]
        labels = [dp.label_ignore for dp in minibatch]
        
        # --- augmentations ---
        # since we perform rotation, we first crop a bigger patch 
        f = []
        s = int(np.sqrt(self.crop_size[0]**2+self.crop_size[1]**2))
        f.append( lambda sample: nt.crop(sample, (s,s), "random") )
        f.append( lambda sample: nt.random_rotation_2d(sample, range=(-180,180)) )
        f.append( lambda sample: nt.crop(sample, self.crop_size, "center") )        
        f.append( lambda sample: nt.random_flip(sample, axis=(0,1), p=(0.5,0.5)) ) 
        images, labels = nt.process_in_batch(f, images, labels)
        
        f = lambda image: nt.random_intensity_remap(image, 0.1)
        images = nt.process_in_batch(f, images) 
        # --- augmentations ---
        
        images  = np.transpose(np.float32(images)/255.0, (0, 3, 1, 2))
        labels  = np.int64(labels)
        weights = None
        
        images  = nt.to_torch(images).cuda()               
        labels  = nt.to_torch(labels).cuda()  
        if weights is not None:
            weights = nt.to_torch(weights).cuda()

        pred = network(images)
        
        loss = loss_function(pred, labels, weights)        
        loss_v = float(nt.from_torch(loss))

        if np.isnan(loss_v) or np.isinf(loss_v):
            return {"loss": loss_v,
                    "pred": nt.from_torch(pred),
                    "labels": nt.from_torch(labels)}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        if lr_scheduler is not None:
            lr_scheduler.step()      
            
        if iterations%self.retrace_every==0 and iterations!=0:            
            self.retracing(iterations, network)
        
        return {"loss": loss_v}
    
class Validation(object):    
    
    def __init__(self, crop_size, margin_size, dataset, out_channels, output_path):    
        
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataset = dataset 
        self.out_channels = out_channels
        self.output_path = output_path
    
    def __call__(self, iteration, network, loss_function):

        losses = []
        scores = {"dice":[], "f1":[], "precision":[], "recall":[],
                  "corr":[], "comp":[], "qual":[]}

        network.train(False)        
        with nt.torch_no_grad:
            for i,dp in enumerate(self.dataset):
                
                image_i = np.transpose(np.float32(dp.image[None])/255.0, (0, 3, 1, 2))
                label_i = np.int64(dp.label[None])
                weights_i = None
                
                image_i  = nt.to_torch(image_i, volatile=True).cuda()               
                label_i  = nt.to_torch(label_i, volatile=True).cuda()
                if weights_i is not None:
                    weights_i = nt.to_torch(weights_i, volatile=True).cuda()                

                out_shape = (image_i.shape[0],self.out_channels,*image_i.shape[2:])
                pred_i = nt.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred_i = nt.process_in_chuncks(image_i, pred_i, 
                                               lambda chunk: network(chunk),
                                               self.crop_size, self.margin_size)                

                loss = loss_function(pred_i, label_i, weights_i)
                loss_v = float(nt.from_torch(loss))   
                losses.append(loss_v)
                
                pred_np = nt.from_torch(pred_i)[0]
                label_np = nt.from_torch(label_i)[0]

                pred_mask = np.argmax(pred_np, axis=0)
                prob = nt.softmax(pred_np)[1]

                dice  = nt.dice_score(pred_mask, label_np)
                pr,re = nt.precision_recall(pred_mask, label_np)
                f1    = nt.f1(pred_mask, label_np)
                
                corr, comp, qual = tl.correctness_completeness_quality(skeletonize(pred_mask>0), 
                                                                       dp.label, slack=5, 
                                                                       eps=1e-12)
                
                scores["dice"].append(dice)
                scores["precision"].append(pr)
                scores["recall"].append(re)
                scores["f1"].append(f1)
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # save the prediction here
                if True:
                    output_valid = os.path.join(self.output_path, "output_valid")
                    nt.mkdir(output_valid)
                    tifffile.imsave(os.path.join(output_valid, "prob_{:06d}_iter{:06d}.tiff".format(i,iteration)), prob)
                
        network.train(True)

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean dice-score: {:0.3f}".format(np.mean(scores["dice"])))   
        logger.info("\tMean prec/rec: {:0.3f}/{:0.3f}".format(np.mean(scores["precision"]), np.mean(scores["recall"])))  
        logger.info("\tMean f1: {:0.3f}".format(np.mean(scores["f1"])))  
        logger.info("\tMean corr: {:0.3f}".format(np.mean(scores["corr"])))
        logger.info("\tMean comp: {:0.3f}".format(np.mean(scores["comp"])))
        logger.info("\tMean qual: {:0.3f}".format(np.mean(scores["qual"])))        
        logger.info("\tPercentile losses: {}".format(np.percentile(losses, [0, 25, 50, 75, 100])))
        logger.info("\tPercentile f1: {}".format(np.percentile(scores["f1"], [0, 25, 50, 75, 100])))        

        return {"loss": np.mean(losses),
                "mean_dice": np.mean(scores["dice"]),
                "mean_precision": np.mean(scores["precision"]),
                "mean_recall": np.mean(scores["recall"]),
                "mean_f1": np.mean(scores["f1"]),
                "mean_corr": np.mean(scores["corr"]),
                "mean_comp": np.mean(scores["comp"]),
                "mean_qual": np.mean(scores["qual"]),                
                "losses": np.float_(losses),
                "scores": scores}

def process_dataset(dp, in_channels):
    if in_channels<=1:
        image = dp.image[:,:,None] # adding one dimension 
    else:
        image = dp.image
        
    '''
    # adding an ignore region
    dilated = binary_dilation(dp.label_thin.copy(), iterations=1)
    label_ignore = np.zeros_like(dilated, np.uint8)
    label_ignore[dilated>0] = IGNORE
    label_ignore[dp.label_thin.copy()>0] = 1        
    '''
    label = dp.label_thin
    label_ignore = dp.label_thin
    weights = np.ones_like(dp.label_thin, np.float32)
    
    return ExtDataPoint(image, label, label_ignore, weights, dp.node_coords, dp.edges)

def main(config_file="main.config"):
    
    torch.set_num_threads(1)
    #cv2.setNumThreads(8)    
    
    __c__ = nt.yaml_read(config_file)
    output_path = __c__["output_path"]
    dataset = __c__["dataset"]
    batch_size = __c__["batch_size"]
    
    nt.mkdir(output_path)
    nt.config_logger(os.path.join(output_path, "main.log"))    
    
    copyfile(config_file, os.path.join(output_path, "main.config"))
    copyfile(__file__, os.path.join(output_path, "main.py"))

    logger.info("Command line: {}".format(' '.join(sys.argv)))

    logger.info("Loading training dataset '{}'...".format(dataset))
    dataset_training = load_dataset(dataset, "training", size="orig", labels="thin", graph=True, each=1)
    logger.info("Done. {} datapoints loaded.".format(len(dataset_training)))

    logger.info("Loading validation dataset '{}'...".format(dataset))
    dataset_validation = load_dataset(dataset, "testing", size="orig", labels="thin", each=1)
    logger.info("Done. {} datapoints loaded.".format(len(dataset_validation)))  

    logger.info("Generating labels...")
    f = lambda dp: process_dataset(dp, __c__["in_channels"])
    dataset_training = [f(dp) for dp in dataset_training]
    dataset_validation = [f(dp) for dp in dataset_validation]
            
    class_weights = compute_class_weights((dp.label[dp.label!=IGNORE] for dp in dataset_training), 
                                          __c__["num_classes"])
    logger.info("Class weights {}.".format(class_weights))                            
        
    retracing = Retracing(tuple(__c__["crop_size_test"]),
                          tuple(__c__["margin_size"]),
                          dataset_training,
                          __c__["num_classes"],
                          output_path,
                          __c__["rad"],
                          __c__["alpha"],
                          __c__["dilation"])        
    training_step = TrainingStep(batch_size, 
                                 tuple(__c__["crop_size"]),
                                 dataset_training,
                                 None,
                                 __c__["retrace_every"],
                                 retracing)
    validation = Validation(tuple(__c__["crop_size_test"]),
                            tuple(__c__["margin_size"]),
                            dataset_validation,
                            __c__["num_classes"],
                            output_path) 

    logger.info("Creating model...")
    network = UNet(in_channels=__c__["in_channels"], 
                   m_channels=__c__["m_channels"], 
                   out_channels=__c__["num_classes"], 
                   n_convs=__c__["n_convs"], 
                   n_levels=__c__["n_levels"], 
                   dropout=__c__["dropout"], 
                   batch_norm=__c__["batch_norm"], 
                   upsampling=__c__["upsampling"],
                   pooling=__c__["pooling"],
                   three_dimensional=__c__["three_dimensional"]).cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=__c__["lr"], 
                                 weight_decay=__c__["weight_decay"])
   
    if __c__["lr_decay"]:
        lr_lambda = lambda it: 1/(1+it*__c__["lr_decay_factor"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None
    
    loss_function = nt.CrossEntropyLoss(class_weights, ignore_index=IGNORE).cuda()

    logger.info("Running...")
    trainer = nt.Trainer(training_step=lambda iter: training_step(iter, network, optimizer, 
                                                                lr_scheduler, loss_function), 
                         validation   =lambda iter: validation(iter, network, loss_function), 
                         valid_every=__c__["valid_every"],
                         print_every=__c__["print_every"], 
                         save_every=__c__["save_every"], 
                         save_path=output_path, 
                         save_objects={"network":network},
                         save_callback=None)    

    trainer.run_for(__c__["num_iters"])

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()   
    parser.add_argument("--config_file", "-c", type=str, default="config.config")

    args = parser.parse_args()

    main(**vars(args))

# CUDA_VISIBLE_DEVICES=3 python main_retracing_2d.py -c main_retracing_2d.config