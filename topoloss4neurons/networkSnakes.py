import numpy as np
from matplotlib import pyplot as plt
import cv2
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

def internal_neighbour_force_3D(s):
    in_forces = np.zeros((len(s), 3))
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
                    print("Problem neigh1")
                    
                if neigh2.neighbours[0] == n.index:
                    neigh22 = s.getindex(neigh2.neighbours[1])
                elif neigh2.neighbours[1] == n.index:
                    neigh22 = s.getindex(neigh2.neighbours[0])
                else:
                    print("Problem neigh2")
                
                f_smo = s.beta * ((neigh11.location - neigh1.location*2 + n.location) - 
                                  2*(neigh1.location - n.location*2 + neigh2.location) + 
                                  (n.location - neigh2.location*2 + neigh22.location))
            in_forces[i,:] = -f_rig - f_smo
    return in_forces

def force_3D_grad(s, alpha = 1):
    locs = np.array([n.location for n in s])
    force = s.surface_dist[locs.astype(np.uint16)[:,0], locs.astype(np.uint16)[:,1], locs.astype(np.uint16)[:,2]]
    direction_phi = s.surface_phi[locs.astype(np.uint16)[:,0], locs.astype(np.uint16)[:,1], locs.astype(np.uint16)[:,2]]
    direction_theta = s.surface_theta[locs.astype(np.uint16)[:,0], locs.astype(np.uint16)[:,1], locs.astype(np.uint16)[:,2]]
    fx = force * np.sin(direction_theta) * np.cos(direction_phi + np.pi)
    fy = force * np.sin(direction_theta) * np.sin(direction_phi + np.pi)
    fz = force * np.cos(direction_theta + np.pi)
    return alpha*np.array([fz, fy, fx]).T

class node_3D:
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
    
    def update(self, f, xmax, ymax, zmax):
        if self.location[0] > 2 and self.location[0] < xmax-2:
            self.location[0] += f[0]
        if self.location[1] > 2 and self.location[1] < ymax-2:
            self.location[1] += f[1]
        if self.location[2] > 2 and self.location[2] < zmax-2:
            self.location[2] += f[2]

        self.location[0] = np.clip(self.location[0], 0, xmax)
        self.location[1] = np.clip(self.location[1], 0, ymax)
        self.location[2] = np.clip(self.location[2], 0, zmax)

class snakes_3D:
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
        dx = convolve(surface, np.array([[[1/2, 0, -1/2]]]))
        dy = convolve(surface, np.array([[[1/2, 0, -1/2]]]).transpose((1,2,0)))
        dz = convolve(surface, np.array([[[1/2, 0, -1/2]]]).transpose((2,1,0)))
        self.surface_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        self.surface_phi = np.arctan2(dy,dx)
        self.surface_theta = np.arctan2(np.sqrt(dx**2 + dy**2),dz)
        
    def update(self, iterations, extraFs = 0, alphaext = 0.2):
        forces = np.zeros((len(self.nodes), 3))
        for i in range(iterations):
            forces = internal_neighbour_force_3D(self) + force_3D_grad(self, alphaext) + extraFs
        return forces

class network_snakes_3D:
    def __init__(self, snakes, networkPoints):
        self.snakes = snakes
        self.networkPoints = networkPoints
        self.netPointSnakes = {}
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
        fig = plt.figure(figsize=(32,8))
        ax = fig.add_subplot(141)
        ax.imshow(self.surface.min(2))
        ax3 = fig.add_subplot(142)
        ax3.imshow(self.surface.min(1))
        ax4 = fig.add_subplot(143)
        ax4.imshow(self.surface.min(0))
        ax2 = fig.add_subplot(144, projection='3d')
        for s in self.snakes:
            locs = np.array([n.location for n in s])

            ax.scatter(locs[:,1], locs[:,0], c="r", s=size)
            ax.plot(locs[:,1], locs[:,0], c="r")

            ax3.scatter(locs[:,2], locs[:,0], c="r", s=size)
            ax3.plot(locs[:,2], locs[:,0], c="r")

            ax4.scatter(locs[:,2], locs[:,1], c="r", s=size)
            ax4.plot(locs[:,2], locs[:,1], c="r")

            ax2.scatter(locs[:,2], locs[:,1], locs[:,0], s=size)
            ax2.plot(locs[:,2], locs[:,1], locs[:,0])

        fig.show()
    
    def get_gt(self):
        new_gt = np.zeros((self.surface.shape[0],self.surface.shape[1],self.surface.shape[2]))
        renderSnakes2lbl(self, new_gt)
        new_gt = skeletonize(new_gt)
        new_gt = distance_transform_edt(~new_gt)
        new_gt[new_gt>15] = 15
        return new_gt
    
    def update(self, iterations, alphaext=0.2, alphatop=0.2, verbose=30, size=1):
        for i in range(iterations):
            if (i+1) % verbose == 0:
                self.visualize(size=size)
            forces = []
            for i,s in enumerate(self.snakes):
                sna = self.snakes.copy()
                sna.remove(s)
                forces.append(s.update(1, 0, alphaext))
            
            for k, s in enumerate(self.snakes):
                for j, n in enumerate(s.nodes):
                    n.update(forces[k][j], self.surface.shape[0] - 1, self.surface.shape[1] - 1, self.surface.shape[2] - 1)
            
def network_plot_3D(G, angle, save=False):
    from mpl_toolkits.mplot3d import Axes3D
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, s=2, edgecolors='k', alpha=0.7)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
    
    # Set the initial view
    ax.view_init(10, angle)

    # Hide the axes
#     ax.set_axis_off()

    if save is not False:
#         plt.savefig("C:\scratch\\data\ " + str(angle).zfill(3)+".png")
        plt.close('all')
    else:
        plt.show()
    
    return

def traceLine(lbl,begPoint,endPoint):
    d=endPoint-begPoint
    s=begPoint
    mi=np.argmax(np.fabs(d))
    coef=d/d[mi]
    sz=np.array(lbl.shape)
    numsteps=int(abs(d[mi]))+1
    step=int(d[mi]/abs(d[mi]))
    for t in range(0,numsteps):
        pos=np.fabs(s+coef*t*step)
        if np.all(pos<sz) and np.all(pos>=0):
            #print(pos)
            lbl[tuple(pos.astype(np.int))]=1
    return lbl

def renderSnakes2lbl(ns, lbl_r):
    for s in ns.snakes:
        nodes = s.nodes
        for i in range(len(nodes)-1):
            x,y,z = nodes[i].location
            xp,yp,zp = nodes[i+1].location
            if (x!=xp or y!=yp or z!=zp):
                traceLine(lbl_r,np.array([x,y,z]),np.array([xp,yp,zp]))
