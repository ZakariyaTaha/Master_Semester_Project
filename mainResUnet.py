import os
import sys
import logging
import argparse
import itertools
import numpy as np
from copy import deepcopy
from shutil import copyfile
from collections import namedtuple
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import skeletonize_3d
import torch
from networktraining import utils
import networktraining as nt
from graph_from_skeleton.graph_from_skeleton import graph_from_skeleton, make_graph
import topoloss4neurons
from topoloss4neurons.networksResUnet import ResUNet
from topoloss4neurons import load_dataset
import gradImSnake
from scipy.ndimage.morphology import distance_transform_edt
from topoloss4neurons.scores.toolong_tooshort_score import find_connectivity_3d, create_graph_3d, extract_gt_paths, toolong_tooshort_score
import pickle
from loss_MSE_GaussSnake_wGrad_v1 import Loss_MSE_GaussSnake_wGrad

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

logger = logging.getLogger(__name__)

ExtDataPoint = namedtuple("ExtDataPoint", ['image', 'label', 'graph'])

pathss = pickle_read("/cvlabdata2/home/oner/Snakes/bbp_paths/paths_2long_2short_neurons.pickle")

class TrainingStep(object):

    def __init__(self, batch_size, crop_size, dataset, probas, snakes, snakes_start=0):
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.dataset = dataset
        self.probas = probas
        self.snakes = snakes

        self.snakes_start = snakes_start

    def __call__(self, iterations, network, optimizer, lr_scheduler, loss_function, loss_function2):

        replace = True if len(self.dataset)<self.batch_size else False
        minibatch = [self.dataset[i] for i in np.random.choice(np.arange(len(self.dataset)),
                                                               self.batch_size,
                                                               replace=replace,
                                                               p=self.probas)]

        images = [dp.image.copy() for dp in minibatch]
        labels = [dp.label.copy() for dp in minibatch]
        graphs = [dp.graph for dp in minibatch]
        # --- augmentation ---
        f = []
        f.append( lambda sample: nt.crop(sample, self.crop_size, "random") )
#         f.append( lambda sample: nt.random_flip(sample, axis=(0,1,2), p=(0.5,0.5,0.5)) )
        images, labels, slices = nt.process_in_batch(f, images, labels)

        images  = np.transpose(np.float32(images), (0,4,1,2,3))
        labels  = np.transpose(np.float32(labels), (0,4,1,2,3))

        images  = nt.to_torch(images).cuda()

        preds = network(images)
        if self.snakes and (iterations >= self.snakes_start):
            loss = loss_function(preds, graphs, slices)
        else:
            labels  = nt.to_torch(labels).cuda()
            loss = loss_function2(preds, labels)
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

        return {"loss": float(loss_v)}

class Validation(object):

    def __init__(self, crop_size, margin_size, dataset_val, dataset_train, out_channels, output_path, metric_start):
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataset_val = dataset_val
        self.dataset_train = dataset_train
        self.out_channels = out_channels
        self.output_path = output_path
        self.metric_start = metric_start

        self.bestf1 = 0
        self.bestpre = 0
        self.bestrec = 0
        self.bestqual = 0
        self.bestcomp = 0
        self.bestcorr = 0
        self.bestapls = 0
        self.besttcor = 0
        self.bestf1_iter = 0
        self.bestpre_iter = 0
        self.bestrec_iter = 0
        self.bestqual_iter = 0
        self.bestcomp_iter = 0
        self.bestcorr_iter = 0
        self.bestapls_iter = 0
        self.besttcor_iter = 0
        self.bestth_f1 = 0
        self.bestth_qual = 0
        self.bestth_apls = 0
        self.bestth_tcor = 0

    def __call__(self, iteration, network, loss_function):

        losses = []
        preds = []
        scores = {"dice":[], "f1":[], "precision":[], "recall":[],
                  "corr":[], "comp":[], "qual":[], "apls":[], "tcor":[], 
                  "tlng":[], "tshr":[], "tinf":[]}

        network.train(False)
        with nt.torch_no_grad:
            for i,dp in enumerate(self.dataset_val):

                image_i = np.transpose(np.float32(dp.image[None]), (0,4,1,2,3))
                label_i = np.transpose(np.float32(dp.label[None]), (0,4,1,2,3))

                image_i  = nt.to_torch(image_i, volatile=True).cuda()
                label_i  = nt.to_torch(label_i, volatile=True).cuda()

                out_shape = (image_i.shape[0],self.out_channels,*image_i.shape[2:])
                pred_i = nt.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred_i = nt.process_in_chuncks(image_i, pred_i,
                                               lambda chunk: network(chunk),
                                               self.crop_size, self.margin_size)

                loss = loss_function(pred_i, label_i)
                loss_v = float(nt.from_torch(loss))
                losses.append(loss_v)

                pred_np = nt.from_torch(pred_i)[0]
                preds.append(pred_np)
                label_np = nt.from_torch(label_i)[0]
                ths = [3]
                dices = np.zeros((len(ths)))
                pres = np.zeros((len(ths)))
                recs = np.zeros((len(ths)))
                f1s = np.zeros((len(ths)))
                corrs = np.zeros((len(ths)))
                comps = np.zeros((len(ths)))
                quals = np.zeros((len(ths)))
                aplss = np.zeros((len(ths)))
                tcors = np.zeros((len(ths)))
                tshrs = np.zeros((len(ths)))
                tlngs = np.zeros((len(ths)))
                tinfs = np.zeros((len(ths)))
                
                apls = 0
                correct = 0
                tooshort = 0
                toolong = 0
                infeasible = 0
                for ti, th in enumerate(ths):
                    pred_mask = (pred_np < th)[0] #skeletonize_3d((pred_np < th)[0])//255
                    label_mask = (label_np==0)

                    dice = nt.dice_score(pred_mask, label_mask)
                    pre,rec = nt.precision_recall(pred_mask, label_mask)
                    f1    = nt.f1(pred_mask, label_mask)
                    if iteration >= self.metric_start:
                        graph_pred = graph_from_skeleton(pred_mask, angle_range=(170,190), dist_line=0.1, dist_node=2, verbose=False)
                        graph_pred = make_graph(graph_pred, False)
                        try:
                            apls = nt.apls(graph_pred, dp.graph)
                        except:
                            print("APLS Error")
                            apls = 0
                        try:
                            graph_pred_t = create_graph_3d(pred_mask)

                            tot,correct,tooshort,toolong,infeasible,res = toolong_tooshort_score(pathss[i], 
                                                                                             graph_pred_t, 
                                                                                             radius_match=5, 
                                                                                             length_deviation=0.15)
                            print(correct,tooshort,toolong,infeasible)
                        except:
                            correct = 0
                            tooshort = 0
                            toolong = 0
                            infeasible = 0
                            print("TLTS Error")
                    corr, comp, qual = topoloss4neurons.correctness_completeness_quality(pred_mask, label_mask, slack=3)

                    dices[ti] = dice
                    pres[ti] = pre
                    recs[ti] = rec
                    f1s[ti] = f1
                    corrs[ti] = corr
                    comps[ti] = comp
                    quals[ti] = qual
                    aplss[ti] = apls
                    tcors[ti] = correct
                    tshrs[ti] = tooshort
                    tlngs[ti] = toolong
                    tinfs[ti] = infeasible


                scores["dice"].append(dices)
                scores["precision"].append(pres)
                scores["recall"].append(recs)
                scores["f1"].append(f1s)
                scores["corr"].append(corrs)
                scores["comp"].append(comps)
                scores["qual"].append(quals)
                scores["apls"].append(aplss)
                scores["tcor"].append(tcors)
                scores["tshr"].append(tshrs)
                scores["tlng"].append(tlngs)
                scores["tinf"].append(tinfs)

                # save the prediction here
                output_valid = os.path.join(self.output_path, "output_valid")
                nt.mkdir(output_valid)
                np.save(os.path.join(output_valid, "pred_{:06d}_final.npy".format(i,iteration)), pred_np)

        scores["qual"] = np.nan_to_num(scores["qual"])
        scores["apls"] = np.nan_to_num(scores["apls"])
        f1_total = np.mean(scores["f1"],axis=0)
        pre_total = np.mean(scores["precision"],axis=0)
        rec_total = np.mean(scores["recall"],axis=0)
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)
        apls_total = np.mean(scores["apls"],axis=0)
        tcor_total = np.mean(scores["tcor"],axis=0)

        f1_max = np.nanargmax(f1_total)
        qual_max = np.nanargmax(qual_total)
        apls_max = np.nanargmax(apls_total)
        tcor_max = np.nanargmax(tcor_total)

        if self.bestf1 < f1_total[f1_max]:
            self.bestf1 = f1_total[f1_max]
            self.bestth_f1 = ths[f1_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestf1.npy".format(i,iteration)), preds[i])
            self.bestf1_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestf1.pickle"),
                             network.state_dict())
        if self.bestrec < rec_total[f1_max]:
            self.bestrec = rec_total[f1_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestrec.npy".format(i,iteration)), preds[i])
            self.bestrec_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestrec.pickle"),
                             network.state_dict())
        if self.bestpre < pre_total[f1_max]:
            self.bestpre = pre_total[f1_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestpre.npy".format(i,iteration)), preds[i])
            self.bestpre_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestpre.pickle"),
                             network.state_dict())
        if self.bestqual < qual_total[qual_max]:
            self.bestqual = qual_total[qual_max]
            self.bestth_qual = ths[qual_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestqual.npy".format(i,iteration)), preds[i])
            self.bestqual_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestqual.pickle"),
                             network.state_dict())
        if self.bestcomp < comp_total[qual_max]:
            self.bestcomp = comp_total[qual_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestcomp.npy".format(i,iteration)), preds[i])
            self.bestcomp_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestcomp.pickle"),
                             network.state_dict())
        if self.bestcorr < corr_total[qual_max]:
            self.bestcorr = corr_total[qual_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestcorr.npy".format(i,iteration)), preds[i])
            self.bestcorr_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestcorr.pickle"),
                             network.state_dict())
        if self.bestapls < apls_total[apls_max]:
            self.bestapls = apls_total[apls_max]
            self.bestth_apls = ths[apls_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestapls.npy".format(i,iteration)), preds[i])
            self.bestapls_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_bestapls.pickle"),
                             network.state_dict())
          
        if self.besttcor < tcor_total[tcor_max]:
            self.besttcor = tcor_total[tcor_max]
            self.bestth_tcor = ths[tcor_max]
            for i in range(len(self.dataset_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_besttcor.npy".format(i,iteration)), preds[i])
            self.besttcor_iter = iteration
            utils.torch_save(os.path.join(self.output_path, "network_besttcor.pickle"),
                             network.state_dict())

        file = open(os.path.join(self.output_path,"bestResults.txt"),"a")

        file.write("Best f1 in iter: {}\n".format(self.bestf1_iter))
        file.write("Best rec in iter: {}\n".format(self.bestrec_iter))
        file.write("Best pre in iter: {}\n".format(self.bestpre_iter))
        file.write("Best qual in iter: {}\n".format(self.bestqual_iter))
        file.write("Best comp in iter: {}\n".format(self.bestcomp_iter))
        file.write("Best corr in iter: {}\n".format(self.bestcorr_iter))
        file.write("Best apls in iter: {}\n".format(self.bestapls_iter))
        file.write("Best tcor in iter: {}\n".format(self.besttcor_iter))

        file.write("*"*25)
        file.write("\n")
        file.write("*"*25)
        file.write("\n")

        file.close()

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean dice-score: {:0.3f}".format(np.mean(scores["dice"])))
        logger.info("\tMean prec/rec: {:0.3f}/{:0.3f}".format(np.mean(scores["precision"]), np.mean(scores["recall"])))
        logger.info("\tMean f1: {:0.3f}".format(f1_total[f1_max]))
        logger.info("\tMean corr: {:0.3f}".format(np.mean(scores["corr"])))
        logger.info("\tMean comp: {:0.3f}".format(np.mean(scores["comp"])))
        logger.info("\tMean qual: {:0.3f}".format(qual_total[qual_max]))
        logger.info("\tMean apls: {:0.3f}".format(apls_total[apls_max]))
        logger.info("\tMean tcor: {:0.3f}".format(tcor_total[tcor_max]))
        logger.info("\tPercentile losses: {}".format(np.percentile(losses, [0, 25, 50, 75, 100])))
        logger.info("\tPercentile f1: {}".format(np.percentile(scores["f1"], [0, 25, 50, 75, 100])))
        logger.info("\tPercentile qual: {}".format(np.percentile(qual_total, [0, 25, 50, 75, 100])))

        logger.info("Best f1 is {} in iter {} with th {}".format(self.bestf1, self.bestf1_iter, self.bestth_f1))
        logger.info("Best pre is {} in iter {}".format(self.bestpre, self.bestpre_iter))
        logger.info("Best rec is {} in iter {}".format(self.bestrec, self.bestrec_iter))
        logger.info("Best qual is {} in iter {} with th {}".format(self.bestqual, self.bestqual_iter, self.bestth_qual))
        logger.info("Best comp is {} in iter {}".format(self.bestcomp, self.bestcomp_iter))
        logger.info("Best corr is {} in iter {}".format(self.bestcorr, self.bestcorr_iter))
        logger.info("Best apls is {} in iter {} with th {}".format(self.bestapls, self.bestapls_iter, self.bestth_apls))
        logger.info("Best tcor is {} in iter {} with th {}".format(self.besttcor, self.besttcor_iter, self.bestth_tcor))
        
        network.train(True)

        return {"loss": np.mean(losses),
                "mean_dice": np.mean(scores["dice"]),
                "mean_precision": pre_total[f1_max],
                "mean_recall": rec_total[f1_max],
                "mean_f1": f1_total[f1_max],
                "mean_corr": corr_total[qual_max],
                "mean_comp": comp_total[qual_max],
                "mean_qual": qual_total[qual_max],
                "mean_apls": apls_total[apls_max],
                "losses": np.float_(losses),
                "scores": scores}

    
def invertCoordinates(g):   
    for n in g.nodes:
        g.nodes[n]["pos"]=g.nodes[n]["pos"][-1::-1]
    return g

def process_dataset(dp, in_channels, gr=False, val=False):
    if in_channels==1:
        image = dp.image[:,:,:,None]
        lbl = dp.dist_labels[:,:,:,None]
        if gr:
            if val:
                graph = make_graph(dp.graph, True)
            else:
                graph = invertCoordinates(dp.graph)
        else:
            graph = None
        
    else:
        image = dp.image

    return ExtDataPoint(image, lbl, graph)

def main(config_file="main.config"):

    torch.set_num_threads(1)
    #cv2.setNumThreads(8)

    __c__ = nt.yaml_read(config_file)
    output_path = __c__["output_path"]
    dataset = __c__["dataset"]
    batch_size = __c__["batch_size"]
    threshold = __c__["threshold"]
    snakes = __c__["snakes"]
    metric_start = __c__["metric_start"]

    nt.mkdir(output_path)
    nt.config_logger(os.path.join(output_path, "main.log"))

    copyfile(config_file, os.path.join(output_path, "main.config"))
    copyfile(__file__, os.path.join(output_path, "main.py"))
    
    print("dirs created")
    
    logger.info("Command line: {}".format(' '.join(sys.argv)))

    logger.info("Loading training dataset '{}'...".format(dataset))
    dataset_training = load_dataset(dataset, "training", size="train", labels="all", each=1, threshold=threshold, graph=snakes, brains=__c__["brains"], clip_value=__c__["clip_value"])
    logger.info("Done. {} datapoints loaded.".format(len(dataset_training)))

    logger.info("Loading validation dataset '{}'...".format(dataset))
    dataset_validation = load_dataset(dataset, "testing", size="test", labels="all", each=1, threshold=threshold, graph=False, brains=__c__["brains"], clip_value=__c__["clip_value"])
    logger.info("Done. {} datapoints loaded.".format(len(dataset_validation)))
    '''
    if batch_size>len(dataset_training):
        new_batch_size = len(dataset_training)
        logger.info("Batch-size is larger than the dataset ({}>{})! We set batch_size from {} to {}".format(batch_size, new_batch_size, batch_size, new_batch_size))
        batch_size = new_batch_size
    '''
    logger.info("Generating labels...")
    ft = lambda dp: process_dataset(dp, __c__["in_channels"], val=False)
    fv = lambda dp: process_dataset(dp, __c__["in_channels"], val=True)
    dataset_training = [ft(dp) for dp in dataset_training]
    dataset_validation = [fv(dp) for dp in dataset_validation]
    
    print("datasets created")
          
    training_step = TrainingStep(batch_size,
                                 tuple(__c__["crop_size"]),
                                 dataset_training,
                                 None,
                                 snakes,
                                 __c__["snakes_start"])
    validation = Validation(tuple(__c__["crop_size_test"]),
                            tuple(__c__["margin_size"]),
                            dataset_validation,
                            dataset_training,
                            __c__["num_classes"],
                            output_path,
                            metric_start)

    logger.info("Creating model...")
    network = ResUNet(in_channels=__c__["in_channels"],
                   m_channels=__c__["m_channels"],
                   out_channels=__c__["num_classes"],
                   n_convs=__c__["n_convs"],
                   n_levels=__c__["n_levels"],
                   dropout=__c__["dropout"],
                   batch_norm=__c__["batch_norm"],
                   upsampling=__c__["upsampling"],
                   pooling=__c__["pooling"],
                   three_dimensional=__c__["three_dimensional"]).cuda()
    
    print("training step, validation and network created")
    

#     network.load_state_dict(torch.load("./brain_logs/log_baseline/network_bestapls.pickle"))
    network.train(True)
    optimizer = torch.optim.Adam(network.parameters(), lr=__c__["lr"],
                                 weight_decay=__c__["weight_decay"])

    if __c__["lr_decay"]:
        lr_lambda = lambda it: 1/(1+it*__c__["lr_decay_factor"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = None
    
    stepsz=0.1
    alpha=0.01
    beta=0.01
    crop=[slice(0,96), slice(0,96), slice(0,96)]
    fltrstdev=0.5
    extparam=1
    nsteps=10
    ndims=3
    cropsz=[32,32,32]
    dmax=15
    maxedgelength=5
    extgradfac=2.0
     
    loss_function=Loss_MSE_GaussSnake_wGrad(stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                                            cropsz,dmax,maxedgelength,extgradfac).cuda()
    
    loss_function2=nt.losses.MSELoss().cuda()

    logger.info("Running...")
    print("losses created")
    
    trainer = nt.Trainer(training_step=lambda iter: training_step(iter, network, optimizer,
                                                                lr_scheduler, loss_function, loss_function2),
                         validation   =lambda iter: validation(iter, network, loss_function2),
                         valid_every=__c__["valid_every"],
                         print_every=__c__["print_every"],
                         save_every=__c__["save_every"],
                         save_path=output_path,
                         save_objects={"network":network, "optimizer":optimizer, "lr_scheduler":lr_scheduler},
                         save_callback=None)
    
    print("trainer created and now will start running")
    
    trainer.run_for(__c__["num_iters"])

if __name__ == "__main__":
    
    print("start")
          
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
    
    print("parsed and now main")
    
    main(**vars(args))

# CUDA_VISIBLE_DEVICES=3 python main_baseline_3d.py -c main_baseline_3d.config

