#%load_ext autoreload
#%autoreload 2
#%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import imageio
from scipy.ndimage import binary_dilation

from utils import *
from graph_from_skeleton import graph_from_skeleton

base = "/cvlabdata2/home/citraro/exps/delin/scores_comparison/agata_results/"
skeleton = imageio.imread(base+"skels2/amsterdam_fc30.png").astype(np.uint8)[:2048, :2048]
skeleton_dil = binary_dilation(skeleton, iterations=5)

G = graph_from_skeleton(skeleton, angle_range=(135,225), dist_line=3, 
                        dist_node=10, verbose=True)

plt.figure()
plt.imshow(skeleton_dil)
plot_graph(G)
plt.show()