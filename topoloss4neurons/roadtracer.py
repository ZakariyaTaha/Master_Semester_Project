from collections import namedtuple
import os
import numpy as np
import imageio
import multiprocessing
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from .. import utils

# ===================================================================================
image_height = 4096
scales = {"orig":1,
          "1024" :1024/image_height}
base = "/cvlabdata2/cvlab/datasets_leo/roadtracer"
base_graphs = "/cvlabdata2/cvlab/datasets_leo/roadtracer/graphs"
path_images={"orig":os.path.join(base, 'images'),
             "1024":os.path.join(base, 'images_1024')}
path_labels={"orig":os.path.join(base, 'labels'),
             "1024":os.path.join(base, 'labels_1024')}
path_labels_thin={"orig":os.path.join(base, 'labels_thin'),
                  "1024":os.path.join(base, 'labels_thin_1024')}
train_names = [line.rstrip('\n') for line in open(os.path.join(base, "train.txt"))]
test_names = [line.rstrip('\n') for line in open(os.path.join(base, "test.txt"))]
sequences = {"training": train_names,
             "testing":  test_names,
             "all":      train_names+test_names}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "basename", "filename", "nodes", "edges"])

def get_graph(city):
    nodes = []
    edges = []
    switch = True
    with open(os.path.join(base_graphs, "{}.graph".format(city)), "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y = line.split(' ')
                nodes.append((float(x),float(y)))
            else:
                idx_node1, idx_node2 = line.split(' ')
                edges.append((int(idx_node1),int(idx_node2)))
    nodes = np.array(nodes)
    edges = np.array(edges)

    return nodes, edges

def _data_point(basename, size="orig", labels="all", graph=False):

    filename = os.path.join(path_images[size], basename)
    image = imageio.imread(filename.format("sat"))

    if labels=="orig" or labels=="all":
        filename = os.path.join(path_labels[size], basename)
        label = imageio.imread(filename.format("osm"))
        label = np.uint8(label>128)
    else:
        label = None

    if labels=="thin" or labels=="all":
        filename = os.path.join(path_labels_thin[size], basename)
        label_thin = imageio.imread(filename.format('osm'))
        label_thin = np.uint8(label_thin>128)
    else:
        label_thin = None

    if graph:
        nodes, edges = get_graph(basename.split('_')[0])

        # registration w.r.t the current image
        dx = int(basename.split('_')[1])
        dy = int(basename.split('_')[2])
        nodes[:,0] = nodes[:,0]-dx*4096
        nodes[:,1] = nodes[:,1]-dy*4096
    else:
        nodes = None
        edges = None

    return DataPoint(image, label, label_thin, basename, filename, nodes, edges)

def compute_probas(names):
    samples_name = [name.split('_')[0] for name in names]
    cities = list(set(samples_name))
    invfreq_lt = [1/samples_name.count(c) for c in cities]
    summ = sum(invfreq_lt)
    invfreq_lt = [x/summ for x in invfreq_lt]
    invfreq_lt = dict(zip(cities, invfreq_lt))
    probs = [invfreq_lt[s] for s in samples_name]

    # probs must sum to 1
    sum_probs = sum(probs)
    probs = [p/sum_probs for p in probs]
    return probs

def load_dataset(sequence='training', size="orig", labels="all",
                 each=1, graph=False, probas=True, threads=12):

    names = sequences[sequence][::each]
    if threads<=1:
        data_points = tuple(_data_point(name, size, labels, graph) for name in names)
    else:
        pool = multiprocessing.Pool(threads)
        res = pool.starmap(_data_point,
                           itertools.product(names, [size], [labels], [graph]))
        pool.close()
        pool.join()
        data_points = tuple(res)

    if probas:
        probs = compute_probas(names)
        return data_points, probs

    return data_points

class Roadtracer(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sequence='training', size="orig", use_thin_labels=False,
                 each=1, probas=True, transform=None, seed=9999, oneshot=False):

        self.size = size
        self.use_thin_labels = use_thin_labels
        self.transform = transform
        self.oneshot = oneshot
        self.names = sequences[sequence][::each]
        if probas:
            self.probas = compute_probas(self.names)
        else:
            self.probas = None

        np.random.seed(seed)

        self.lenght = len(self.names) if oneshot else 9999999999

    def __len__(self):
        return self.lenght

    def __getitem__(self, idx):

        if self.oneshot:
            name = self.names[idx]
        else:
            name = np.random.choice(self.names, 1, p=self.probas)[0]

        if not self.use_thin_labels:
            dp = _data_point(name, self.size, "orig")
            image, label = dp.image, dp.label
        else:
            dp = _data_point(name, self.size, "thin")
            image, label = dp.image, dp.thin_label

        if self.transform is not None:
            image, label = self.transform(image, label)

        image, label = np.float32(image)/255.0, np.int64(label)
        image = np.transpose(image, (2,0,1))

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())
