import numpy as np
import sys
import os
import imageio
import pickle
from skimage.morphology import skeletonize_3d, binary_dilation
# sys.path.append("/home/citraro/projects/topoloss4neurons/topo_score")
from toolong_tooshort_score import find_connectivity_3d, create_graph_3d, extract_gt_paths, toolong_tooshort_score

# folder_gt = "/cvlabdata2/home/oner/Snakes/mra/MRAdata_py/dist_lbl_cropped_x2/"
# folder_preds = ["/cvlabdata2/home/oner/Snakes/mra/MRAdata_py/dist_lbl_cropped_x2/"]
# folder_gt = "/cvlabsrc1/home/oner/tabea_dataset/test_dist_label/"
# folder_preds = ["/cvlabsrc1/home/oner/tabea_dataset/test_dist_label/"]

folder_gt = "/cvlabdata2/home/oner/Snakes/codes/Synth/dist_labels/"
folder_preds = ["/cvlabdata2/home/oner/Snakes/codes/Synth/dist_labels/"]

# folder_preds = ["/cvlabdata2/home/oner/Snakes/codes/brain_logs/log_ns_3d_3steps_2conv_pod_apls/output_valid/",
#                "/cvlabdata2/home/oner/Snakes/codes/brain_logs/log_ns_3d_3steps_2conv_pod_apls2/output_valid/",
#                "/cvlabdata2/home/oner/Snakes/codes/brain_logs/log_ns_3d_3steps_2conv_pod_apls3/output_valid/",
#                "/cvlabdata2/home/oner/Snakes/codes/brain_logs/log_ns_3d_3steps_2conv_pod_apls4/output_valid/",
#                "/cvlabdata2/home/oner/Snakes/codes/brain_logs/log_ns_3d_3steps_2conv_pod_apls5_nobeta2/output_valid/"]
#folder_pred = "/cvlabdata2/home/kozinski/topoLoss/log_retrace_v5_reweighting_only/last_net_test_set_1"
filename_results = os.path.join(folder_preds[0].split('/')[-2]+"_"+folder_preds[0].split('/')[-1]+"_2long_2short_results.txt")
load_saved_paths = True

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

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data
        
def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def sigmoid(x):
    e = np.exp(x)
    return e/(e + 1)

def load_pred(filename, th=1):
    volume_pred = np.load(filename)<th
    volume_pred_s = skeletonize_3d(volume_pred.copy())>128
    return volume_pred_s

def load_gt(filename):
    volume_gt = np.load(filename)==0
#     volume_gt = volume_gt-1
#     volume_gt[volume_gt==255] = 0
    volume_gt = volume_gt>0

    volume_gt_s = skeletonize_3d(volume_gt.copy())>128
    return volume_gt_s

print("="*20)
print(folder_preds)

filename_gts = ["data_{}.npy".format(i) for i in range(20,30)]
print("filename_gts:")
for f in filename_gts:
    print(f)

filename_predss = [["data_{}.npy".format(i) for i in range(20,30)]]
print("filename_preds:")
for f in filename_predss:
    print(f)
    
#mkdir(output)
                                   
                                    
if load_saved_paths:
    pathss = pickle_read("/cvlabdata2/home/oner/Snakes/codes/paths_2long_2short_synth.pickle")
else:
    pathss = []
    for fgt in filename_gts:
        volume_gt_s = load_gt(os.path.join(folder_gt, fgt))
        graph_gt = create_graph_3d(volume_gt_s)    
    
        paths = extract_gt_paths(graph_gt, 
                                 N=2000, 
                                 min_path_length=50)
        print("n_paths_gt", len(paths), "mean_length", np.mean([len(x["shortest_path_gt"]) for x in paths]))        
        pathss.append(paths)       
    
    pickle_write("/cvlabdata2/home/oner/Snakes/codes/paths_2long_2short_synth.pickle",pathss)
#     pickle_write()
    
print("-->File with results: ", filename_results)    
    
file = open(filename_results, "w")    
file.write("threshold correct tooshort toolong infeasible\n")
file.close()
file = open(filename_results, "a")

for folder_pred in folder_preds:
    print(folder_pred)
    for filename_preds in filename_predss:
        for th in [2,3]:    
            ps = []
            for i,(paths, fpred) in enumerate(zip(pathss, filename_preds)):

                volume_pred_s = load_pred(os.path.join(folder_pred, fpred), th)

                graph_pred = create_graph_3d(volume_pred_s)

                tot,correct,tooshort,toolong,infeasible,res = toolong_tooshort_score(paths, 
                                                                                     graph_pred, 
                                                                                     radius_match=20, 
                                                                                     length_deviation=0.15)

                ps.append([correct,tooshort,toolong,infeasible])

                print("th", th, "name",os.path.basename(fpred), "correct", correct, \
                      "tooshort", tooshort, "toolong", toolong, "infeasible", infeasible)

            ps = np.array(ps)

            mean = np.mean(ps, axis=0)
            print("average:")
            print("th", th, "correct", mean[0], "tooshort", mean[1], "toolong", mean[2], "infeasible", mean[3])
            print("----------------------------------")

            file.write("{} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(th, mean[0], mean[1], mean[2], mean[3]))
    
file.close()    