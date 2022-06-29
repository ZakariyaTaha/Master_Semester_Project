
from collections import namedtuple
import os
import numpy as np
import imageio
from scipy.ndimage.morphology import distance_transform_edt as dist
from .. import utils
import pickle
from graph_from_skeleton import graph_from_skeleton
from graph_from_skeleton.utils import *
# ===================================================================================
image_height = 512
scales = {"orig":1,
          "256" :256/image_height}
base = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/orig"
base_graphs = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/graphs_isbi12/lbl_graph/train"
base_ns = "/cvlabdata2/home/oner/Snakes/NetworkSnakes/"
path_images={"orig":os.path.join(base, 'train_images'),
             "256" :os.path.join(base, 'train_images_256')}
path_labels={"orig":os.path.join(base, 'train_labels'),
             "256" :os.path.join(base, 'train_labels_256')}
path_labels_thin={"orig":os.path.join(base, 'train_labels_thin'),
                  "256" :os.path.join(base, 'train_labels_thin_256')}
sequences = {"training": range( 0,15,1),
             "testing":  range(15,30,1),
             "all":      range( 0,30,1)}
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "label", "label_thin", "weights", "basename", "filename", "node_coords", "edges", "ns"])

def _data_point(fid, size="orig", labels="all", dist_lbl=True, graph=False, dist_thresh=20, snakes=True, ix=0):

    basename = 'slice_{:02d}.tiff'.format(fid)
    filename = os.path.join(path_images[size], basename)
    image = imageio.imread(filename)
    weights = None
    if labels=="orig" or labels=="all":
        filename = os.path.join(path_labels[size], basename)
        label = imageio.imread(filename)
        if dist_lbl:
            label = dist(label)
        else:
            label = np.uint8(label<1) # inverting background-foreground
    else:
        label = None

    if labels=="thin" or labels=="all":
        filename = os.path.join(path_labels_thin[size], basename)
        label_thin = imageio.imread(filename)
        if dist_lbl:
            label_thin = dist(label_thin)
            label_thin[label_thin > dist_thresh] = dist_thresh
            # weights = np.ones((label_thin.shape[0], label_thin.shape[1]))
            # weights = 2*dist_thresh - label_thin
            # weights /= weights.mean()
        else:
            label_thin = np.uint8(label_thin<1) # inverting background-foreground
    else:
        label_thin = None

    weights = None

    if graph:
        filename = os.path.join(base_graphs, 'slice_{:02d}.pickle'.format(fid))
        data = utils.pickle_read(filename)
        node_coords = data['node_coordinates']
        edges = data['edges']
    else:
        node_coords = None
        edges = None

    if snakes:
        ns = create_snakes(ix)
    else:
        ns = None

    return DataPoint(image, label, label_thin, weights, basename, filename, node_coords, edges, ns)

def load_dataset(sequence='training', size="orig", labels="all", each=1, dist_lbl=True, graph=False, dist_thresh=20, snakes=True):

    data_points = tuple(_data_point(fid, size, labels, dist_lbl, graph, dist_thresh, snakes, ix) for ix, fid in enumerate(sequences[sequence][::each]))

    return data_points

def create_snakes(ix):
    print(ix)
    base = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/orig/train_labels_thin/slice_{:02d}.tiff".format(ix)
    label_thin = (1-imageio.imread(base))
    graph = graph_from_skeleton.graph_from_skeleton(label_thin, angle_range=(170,190), dist_line=0.1, dist_node=3, verbose=False)
    graph = oversampling_graph(graph,2)
    nodes = []
    for n in graph.nodes(data=True):
        nodes.append([float(n[1]["pos"][1]), float(n[1]["pos"][0])])
    edges = np.array(nx.adjacency_matrix(graph).todense())
    degrees = sum(edges)
    endpoints = np.where(degrees == 1)[0]
    junctions = np.where(degrees > 2)[0]

    neighbours = []
    neighbours2 = []
    alpha = 0.01
    beta = 0.02
    for i in range(len(nodes)):
        ns = list(np.where(edges[i] == 1)[0])
        neighbours.append(ns.copy())
        neighbours2.append(ns.copy())
    # neighbours2 = neighbours.copy()
    # print(neighbours2)
    # For each junction nodes create a snake until you reach a non-2 point
    full_snakes = []
    junction_nodes = []
    junction_inds = []
    for j in junctions:
        while len(neighbours[j]) > 0:
            snake_nodes = []
            if j in junction_inds:
                init_node = junction_nodes[np.where(junction_inds==j)[0][0]]
            else:
                init_node = node(nodes[j], j, neighbours2[j])
                junction_nodes.append(init_node)
                junction_inds.append(j)
            snake_nodes.append(init_node)
            ci = j
            ni = neighbours[ci].pop(0)
            neighbours[ni].remove(ci)
            while degrees[ni] == 2:
                ci = ni
                next_node = node(nodes[ci], ci, neighbours2[ci])
                snake_nodes.append(next_node)
                ni = neighbours[ci].pop(0)
                neighbours[ni].remove(ci)


            next_node = node(nodes[ni], ni, neighbours2[ni])
            snake_nodes.append(next_node)
            full_snakes.append(snakes(snake_nodes, [0,1], alpha, beta))

            if next_node.rank > 2:
                junction_nodes.append(next_node)
                junction_inds.append(ni)


    remaining_nodes = []
    remaining_ends = []
    for i, n in enumerate(neighbours):
        if len(n) == 2:
            remaining_nodes.append(i)
        elif len(n) == 1:
            remaining_ends.append(i)
    # while len(remaining_nodes) > 0:
    for i in remaining_ends:
        if len(neighbours[i]) > 0:
            snake_nodes = []
            init_node = node(nodes[i], i, neighbours2[i])
            snake_nodes.append(init_node)
            ci = i
            ni = neighbours[ci].pop(0)
            neighbours[ni].remove(ci)

            while degrees[ni] == 2:
                ci = ni
                next_node = node(nodes[ci], ci, neighbours2[ci])
                snake_nodes.append(next_node)
                ni = neighbours[ci].pop(0)
                neighbours[ni].remove(ci)
                remaining_nodes.remove(ci)


            next_node = node(nodes[ni], ni, neighbours2[ni])
            snake_nodes.append(next_node)
            full_snakes.append(snakes(snake_nodes, [0,1], alpha=alpha, beta=beta))

    if len(remaining_nodes) > 1:
        for i in remaining_nodes:
            if len(neighbours[i]) > 0:
                snake_nodes = []
                init_node = node(nodes[i], i, neighbours2[i])
                snake_nodes.append(init_node)
                ci = i
                ni = neighbours[ci].pop(0)
                neighbours[ni].remove(ci)

                while degrees[ni] == 2:
                    ci = ni
                    next_node = node(nodes[ci], ci, neighbours2[ci])
                    snake_nodes.append(next_node)
                    try:
                        ni = neighbours[ci].pop(0)
                    except:
                        break
                    neighbours[ni].remove(ci)
                    remaining_nodes.remove(ci)


                next_node = node(nodes[ni], ni, neighbours2[ni])
                snake_nodes.append(next_node)
                full_snakes.append(snakes(snake_nodes, [0,1], alpha=alpha, beta=beta))

    ns = network_snakes(full_snakes,junction_nodes,ix)
    ns.upload_surface(label_thin)
    # with open('./NetworkSnakes/slice_{:02d}.ns'.format(ix), 'wb') as network_s:
    #     pickle.dump(ns, network_s)
    return ns

import numpy as np
from matplotlib import pyplot as plt
#import cv2
from scipy.ndimage.filters import convolve
from scipy.ndimage import distance_transform_edt
from graph_from_skeleton import graph_from_skeleton
import os
import pickle
from topoloss4neurons import utils
from skimage.morphology import skeletonize
from graph_from_skeleton.utils import *
import networkx as nx
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_agg import FigureCanvas
import imageio

def internal_neighbour_force_2D(s):
    in_forces = np.zeros((len(s), 2))
    for i, n in enumerate(s):
        if (n.rank == 2):
            neigh1 = s.getindex(n.neighbours[0])
            neigh2 = s.getindex(n.neighbours[1])
            f_rig = s.alpha * (2*n.location - neigh1.location - neigh2.location)
            f_smo = 0
            if neigh1.rank == 2 and neigh2.rank == 2:
                if neigh1.neighbours[0] == n.index:
                    neigh11 = s.getindex(neigh1.neighbours[1])
                elif neigh1.neighbours[1] == n.index:
                    neigh11 = s.getindex(neigh1.neighbours[0])
                else:
                    pint("Problem neigh1")

                if neigh2.neighbours[0] == n.index:
                    neigh22 = s.getindex(neigh1.neighbours[1])
                elif neigh2.neighbours[1] == n.index:
                    neigh22 = s.getindex(neigh1.neighbours[0])
                else:
                    pint("Problem neigh2")

                f_smo = s.beta * ((neigh11.location - neigh1.location*2 + n.location) -
                                  2*(neigh1.location - n.location*2 + neigh2.location) +
                                  (n.location - neigh2.location*2 + neigh2.location))
            in_forces[i,:] = -f_rig - f_smo
#         else:
#             neigh1 = n.neighbours
#             f_rig = np.zeros(2)
#             f_smo = s.beta * n.rank * n.location
#             neigh2 = []

#             for singleNode in neigh1:
#                 f_smo -= s.beta *singleNode.rank * singleNode.location
#                 neigh2.extend(singleNode.neighbours)
#                 neigh2.remove(n)
#             for singleNode in neigh2:
#                 f_smo += s.beta * singleNode.location
#             in_forces[i,:] = (-f_rig - f_smo)/n.rank

    return in_forces

def force_2D_grad(s, alpha = 1):
    locs = np.array([n.location for n in s])
    force = s.surface_dist[locs.astype(np.uint16)[:,0], locs.astype(np.uint16)[:,1]]
    direction = s.surface_angle[locs.astype(np.uint16)[:,0], locs.astype(np.uint16)[:,1]]
    fx = force * np.cos(direction + np.pi)
    fy = force * np.sin(direction + np.pi)
    return alpha*np.array([fy, fx]).T

class node:
    def __init__(self, location, index, neighbours):
        self.location = np.array(location)
        self.rank = len(neighbours)
        self.index = index
        self.neighbours = neighbours.copy()

    def __getitem__(self, i):
        return self.location[i]

    def __add__(self, n):
        return self.location + n.location

    def __sub__(self, n):
        return self.location - n.location

    def __mul__(self, n):
        return self.location * n

    def update(self, f, xmax, ymax):
        if self.location[0] > 5 and self.location[0] < xmax - 5:
            self.location[0] += f[0]
        if self.location[1] > 5 and self.location[1] < ymax - 5:
            self.location[1] += f[1]

        self.location[0] = np.clip(self.location[0], 0, xmax-1)
        self.location[1] = np.clip(self.location[1], 0, ymax-1)

class snakes:
    def __init__(self, nodes, edges, alpha=0.01, beta=0.02, lr=1):
        self.nodes = nodes
        self.edges = edges
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

    def __getitem__(self, x):
        return self.nodes[x]

    def getindex(self, x):
        for n in self.nodes:
            if n.index == x:
                return n
        return

    def getitems(self):
        return self.nodes.copy()

    def __len__(self):
        return len(self.nodes)

    def upload_surface(self, surface):
        self.surface = surface
        dx = convolve(surface, np.array([[1/2, 0, -1/2]]))
        dy = convolve(surface, np.array([[1/2, 0, -1/2]]).T)

        self.surface_dist = np.sqrt(dx**2 + dy**2)
        self.surface_angle = np.arctan2(dy,dx)

    def update(self, iterations, extraFs = 0, alphaext = 0.2):
        forces = np.zeros((len(self.nodes), 2))
        for i in range(iterations):
            forces = internal_neighbour_force_2D(self) + force_2D_grad(self, alphaext) + extraFs
            return forces

            if i % 10 == 11:
                plt.imshow(self.surface)
                locs = np.array([n.location for n in s])
                plt.scatter(locs[:,1], locs[:,0], c='r')
                plt.plot(locs[:,1], locs[:,0], c='r')
                plt.show()

class network_snakes:
    def __init__(self, snakes, networkPoints, index):
        self.snakes = snakes
        self.networkPoints = networkPoints
        self.netPointSnakes = {}
        self.index = index
        for nP in self.networkPoints :
            nPSn = []
            for sn in snakes :
                if nP in sn.getitems():
                    nPSn.append(sn)
            self.netPointSnakes[nP] = nPSn

        self.num = len(snakes)

    def __getSnakes__(self, networkPoint):
        return self.netPointSnakes[networkPoint]

    def adjSnakes(self, snake):
        snakesPoints = snake.getitems()
        adjSnakes = []
        for nP in self.networkPoints:
            if nP in snakesPoints :
                adjSnakes.extend(self.netPointSnakes[nP])
        if snake in adjSnakes:
            adjSnakes.remove(snake)
        return adjSnakes

    def upload_surface(self, surface):
        self.surface = surface
        for s in self.snakes:
            s.upload_surface(surface)

    def visualize(self, size=2):
        plt.imshow(self.surface)
        for s in self.snakes:
            locs = np.array([n.location for n in s])
            plt.scatter(list(locs[:,1]), list(locs[:,0]), c='r', s=size)
            plt.plot(locs[:,1], locs[:,0], c='r')
        plt.show()

    def get_gt(self):
        plt.figure(figsize=(10,10),dpi=51.2)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(np.zeros((512,512)),cmap="gray")
        for s in self.snakes:
            locs = np.array([n.location for n in s])
            plt.plot(locs[:,1], locs[:,0], c='w')

        plt.savefig("newgt_{}.png".format(self.index), bbox_inches = 'tight',pad_inches = 0)
        plt.close()
        data = imageio.imread("newgt_{}.png".format(self.index))
        new_gt = distance_transform_edt(~skeletonize(np.dot(data[:,:,:3], [0.2989, 0.5870, 0.1140])>10))
        new_gt[new_gt > 20] = 20

        return new_gt

    def update(self, iterations, alphaext=0.2):
        for i in range(iterations):
            forces = []
            for i,s in enumerate(self.snakes):
                sna = self.snakes.copy()
                sna.remove(s)
                forces.append(s.update(1, 0, alphaext))

            for k, s in enumerate(self.snakes):
                for j, n in enumerate(s.nodes):
                    n.update(forces[k][j], self.surface.shape[0] - 1, self.surface.shape[1] - 1)
