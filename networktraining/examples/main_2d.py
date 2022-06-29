import os
import sys
import logging
import argparse
import itertools
import numpy as np
from copy import deepcopy
from shutil import copyfile
from collections import namedtuple
from skimage.external import tifffile
import torch

import networkTraining as nt

import topoloss4neurons
from topoloss4neurons.networks import UNet
from topoloss4neurons import load_dataset

logger = logging.getLogger(__name__)

ExtDataPoint = namedtuple("ExtDataPoint", ['image', 'label'])

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

def crop(images, labels, shape=(224,224), method='random'):
    def _crop_images():
        for image_i, label_i in zip(images, labels):
            yield nt.crop([image_i, label_i], shape=shape, method=method)      

    images, labels = zip(*_crop_images())
    return np.array(images), np.array(labels)

class TrainingStep(object):
    
    def __init__(self, batch_size, crop_size, dataset, probas):
        self.batch_size = batch_size  
        self.crop_size = crop_size
        self.dataset = dataset
        self.probas = probas
        
    def __call__(self, iterations, network, optimizer, lr_scheduler, loss_function):

        minibatch = [self.dataset[i] for i in np.random.choice(np.arange(len(self.dataset)), 
                                                               self.batch_size, 
                                                               replace=False,
                                                               p=self.probas)]

        images = [dp.image for dp in minibatch]
        labels = [dp.label for dp in minibatch]
        
        images, labels = crop(images, labels, self.crop_size, "random")
        
        images  = np.transpose(np.float32(images)/255.0, (0, 3, 1, 2))
        labels  = np.int64(labels)
        weights = None
        
        images  = nt.to_torch(images).cuda()               
        labels  = nt.to_torch(labels).cuda()  
        if weights is not None:
            weights = nt.to_torch(weights).cuda()

        pred = network(images)
        
        loss = loss_function(pred, labels, weights)        
        loss_v = nt.from_torch(loss, True)

        if np.isnan(loss_v) or np.isinf(loss_v):
            return {"loss": loss_v,
                    "pred": nt.from_torch(pred),
                    "labels": nt.from_torch(labels)}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        if lr_scheduler is not None:
            lr_scheduler.step()            
        
        return {"loss": float(loss_v)}
    
class Validation(object):    
    
    def __init__(self, crop_size, margin_size, dataset, out_channels, output_path):        
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataset = dataset 
        self.out_channels = out_channels
        self.output_path = output_path
    
    def __call__(self, iteration, network, loss_function):

        losses = []
        scores = {"dice":[], "f1":[], "precision":[], "recall":[]}
        
        def process_chunk(chunk):
            pred = network(chunk)
            return pred

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
                pred_i = nt.process_in_chuncks(image_i, pred_i, process_chunk, 
                                               self.crop_size, self.margin_size)                

                loss = loss_function(pred_i, label_i, weights_i)
                loss_v = nt.from_torch(loss, True)   
                losses.append(loss_v)
                
                pred_np = nt.from_torch(pred_i)[0]
                label_np = nt.from_torch(label_i)[0]

                pred_mask = np.argmax(pred_np, axis=0)

                dice  = nt.dice_score(pred_mask, label_np)
                pr,re = nt.precision_recall(pred_mask, label_np)
                f1    = nt.f1(pred_mask, label_np)
                
                scores["dice"].append(dice)
                scores["precision"].append(pr)
                scores["recall"].append(re)
                scores["f1"].append(f1)
                
                # save the prediction here
                if True:
                    output_valid = os.path.join(self.output_path, "output_valid")
                    nt.mkdir(output_valid)
                    tifffile.imsave(os.path.join(output_valid, "pred_{}.tiff".format(i)), pred_np)
                
        network.train(True)

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean dice-score: {:0.3f}".format(np.mean(scores["dice"])))   
        logger.info("\tMean prec/rec: {:0.3f}/{:0.3f}".format(np.mean(scores["precision"]), np.mean(scores["recall"])))  
        logger.info("\tMean f1: {:0.3f}".format(np.mean(scores["f1"])))  
        logger.info("\Percentile losses: {}".format(np.percentile(losses, [0, 25, 50, 75, 100])))
        logger.info("\Percentile f1: {}".format(np.percentile(scores["f1"], [0, 25, 50, 75, 100])))        

        return {"loss": np.mean(losses),
                "mean_dice": np.mean(scores["dice"]),
                "mean_precision": np.mean(scores["precision"]),
                "mean_recall": np.mean(scores["recall"]),
                "mean_f1": np.mean(scores["f1"]),
                "losses": np.float_(losses),
                "scores": scores}

def process_dataset(dp, in_channels):
    if in_channels<=1:
        image = dp.image[:,:,None] # adding one dimension 
    return ExtDataPoint(image, dp.label_thin)

def main(config_file="config.config"):
    
    torch.set_num_threads(1)
    #cv2.setNumThreads(8)    
    
    __c__ = nt.yaml_read(config_file)
    output_path = __c__["output_path"]
    dataset = __c__["dataset"]
    
    nt.mkdir(output_path)
    nt.config_logger(os.path.join(output_path, "main.log"))    
    
    copyfile(config_file, os.path.join(output_path, "config.config"))
    copyfile(__file__, os.path.join(output_path, "main.py"))

    logger.info("Command line: {}".format(' '.join(sys.argv)))

    logger.info("Loading training dataset '{}'...".format(dataset))
    dataset_training = load_dataset(dataset, "testing", size="orig", labels="thin", each=1)
    logger.info("Done. {} datapoints loaded.".format(len(dataset_training)))

    logger.info("Loading validation dataset '{}'...".format(dataset))
    dataset_validation = load_dataset(dataset, "testing", size="orig", labels="thin", each=1)
    logger.info("Done. {} datapoints loaded.".format(len(dataset_validation)))

    logger.info("Generating labels...")
    f = lambda dp: process_dataset(dp, __c__["in_channels"])
    dataset_training = [f(dp) for dp in dataset_training]
    dataset_validation = [f(dp) for dp in dataset_validation]
            
    class_weights = compute_class_weights((dp.label for dp in dataset_training), __c__["num_classes"])
                            
    training_step = TrainingStep(__c__["batch_size"], 
                                 tuple(__c__["crop_size"]),
                                 dataset_training,
                                 None)
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
                   pooling=__c__["pooling"]).cuda()

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

# CUDA_VISIBLE_DEVICES=3 python main_baseline_2d.py -c config.config