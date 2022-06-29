import os
import sys
import json
import re
import os
import glob
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))

def json_write(obj, filename):    
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile)
        
def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
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
        
def render_graph(segments, filename, height=3072, width=3072, thickness=4):
    
    mkdir(os.path.dirname(filename))
    
    if isinstance(segments, np.ndarray):
        segments = segments.tolist()
    
    from PIL import Image, ImageDraw

    im = Image.new('RGB', (width, height), (0, 0, 0)) 
    draw = ImageDraw.Draw(im) 
    for p1,p2 in segments:
        draw.line(p1+p2, fill=(255,255,255), width=thickness)
    im.save(filename) 
    
def interpolate_new_nodes(p1, p2, spacing=2):
    
    p1_, p2_ = np.array(p1), np.array(p2)

    segment_length = np.linalg.norm(p1_-p2_)

    new_node_pos = p1_ + (p2_-p1_)*np.linspace(0,1,int(np.ceil(segment_length/spacing)))[1:-1,None]

    return new_node_pos 

def plot_graph(graph, node_size=20, font_size=-1, 
               node_color='y', edge_color='y', 
               linewidths=2, offset=np.array([0,0]), **kwargs):
  
    pos = dict({n:graph.nodes[n]['pos']+offset for n in graph.nodes()})
    nx.draw_networkx(graph, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, font_size=font_size, **kwargs)
    plt.gca().invert_yaxis()
    plt.legend()     
    
def load_graph_txt(filename):
     
    G = nx.Graph()
        
    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y,z = line.split(' ')
                G.add_node(i, pos=(float(x),float(y),float(z)))
                i+=1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1),int(idx_node2))
    
    return G

def save_graph_json(G, filename):
    
    def _tolist(x):
        return x.tolist() if isinstance(x,np.ndarray) else x
    
    mkdir(os.path.dirname(filename))
    
    graph = {"nodes":[int(n) for n in G.nodes()],
             "positions":[_tolist(G.nodes[n]['pos']) for n in G.nodes()],
             "edges":[(int(s),int(t)) for s,t in G.edges()]}
    
    json_write(graph, filename)

def save_graph_txt(G, filename):
    
    mkdir(os.path.dirname(filename))
    
    nodes = list(G.nodes())
    
    file = open(filename, "w+")
    for n in nodes:
        file.write("{:.6f} {:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1], G.nodes[n]['pos'][2]))
    file.write("\r\n")
    for s,t in G.edges():
        file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))
    file.close()

def interpolate_new_nodes(p1, p2, spacing=2):
    
    p1_, p2_ = np.array(p1), np.array(p2)

    segment_length = np.linalg.norm(p1_-p2_)

    new_node_pos = p1_ + (p2_-p1_)*np.linspace(0,1,int(np.ceil(segment_length/spacing)))[1:-1,None]

    return new_node_pos    
    
def oversampling_graph(G, spacing=20):
    edges = list(G.edges())
    for s,t in edges:

        new_nodes_pos = interpolate_new_nodes(G.nodes[s]['pos'], G.nodes[t]['pos'], spacing)

        if len(new_nodes_pos)>0:
            G.remove_edge(s,t)
            n = max(G.nodes())+1

            for i,n_pos in enumerate(new_nodes_pos):
                G.add_node(n+i, pos=tuple(n_pos))

            G.add_edge(s,n)
            for _ in range(len(new_nodes_pos)-1):
                G.add_edge(n,n+1)
                n+=1
            G.add_edge(n,t)
    return G    

def shift_graph(G, shift_x, shift_y):
    H = G.copy()
    for _,data in H.nodes(data=True):
        x,y = data['pos']
        x,y = x+shift_x,y+shift_y
        if isinstance(data['pos'], np.ndarray):
            data['pos'] = np.array([x,y])
        else:
            data['pos'] = (x,y)
    return H 

def crop_graph(G, xmin=None, ymin=None, xmax=None, ymax=None):
    H = G.copy()
    for n in list(H.nodes()):
        p = H.nodes[n]['pos']
        if p[0]>=xmin and p[0]<xmax and p[1]>=ymin and p[1]<ymax:
            pass
        else:
            H.remove_node(n)
    return H