import numpy as np
from skimage import measure
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from munkres import Munkres
from copy import deepcopy
from scipy.spatial.distance import cdist
import time

class TopoGraph(nx.MultiGraph):
    
    def __init__(self, intersections=None, end_of_lines=None, *args, **kwargs):
        super(TopoGraph, self).__init__(*args, **kwargs) 
        
        self.intersections = intersections
        self.end_of_lines = end_of_lines
        
        if intersections is not None and end_of_lines is not None:
            if len(intersections)>0 and len(end_of_lines)>0:
                self.nodes_positions = np.vstack((intersections, end_of_lines))
            elif len(intersections)>0:
                self.nodes_positions = intersections
            elif len(end_of_lines)>0:
                self.nodes_positions = end_of_lines          
        elif intersections is not None:
            self.nodes_positions = intersections
        elif end_of_lines is not None:
            self.nodes_positions = end_of_lines
        else:
            self.nodes_positions = None
            
        self.nodes_indexes = []
            
        # add nodes to the graph
        if self.nodes_positions is not None:
            for i, node in enumerate(self.nodes_positions):
                self.add_node(i, pos=node) 
                self.nodes_indexes.append(i)
                
    def copy(self):
        return deepcopy(self)
    
    def clone(self):
        return self.copy()
    
    

# A biiiiiig list of valid intersections             2 3 4
# These are in the format shown to the right         1 C 5
#                                                    8 7 6 
neighbour = np.array([[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]])
valid_intersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                      [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                      [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                      [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                      [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                      [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                      [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                      [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                      [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                      [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                      [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                      [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                      [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                      [1,0,1,1,0,1,0,0],
                     [0, 1, 0, 0, 1, 1, 1, 0],[0, 0, 1, 1, 1, 0, 0, 1],[1, 1, 1, 0, 0, 1, 0, 0],[1, 0, 0, 1, 0, 0, 1, 1],
                     [0, 0, 1, 1, 1, 0, 1, 0],[1, 1, 1, 0, 1, 0, 0, 0],[1, 0, 1, 0, 0, 0, 1, 1],[1, 0, 0, 0, 1, 1, 1, 0],
                     [1, 1, 1, 0, 0, 0, 0, 1],[1, 0, 0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1, 1, 0],[0, 1, 1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 1, 1, 0, 0],[1, 1, 1, 1, 0, 0, 1, 0],[1, 1, 0, 0, 1, 0, 1, 1],[0, 0, 1, 0, 1, 1, 1, 1]]
valid_end_of_lines = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                      [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
                      
                      [1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],
                      [0,0,1,1,0,0,0,0],[0,0,0,1,1,0,0,0],
                      [0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],
                      [0,0,0,0,0,0,1,1],[1,0,0,0,0,0,0,1]]

def find_closest(points):
    dists = cdist(points, points)
    dists[np.eye(len(points))>0]=np.inf
    idx_min = np.unravel_index(dists.argmin(), dists.shape)
    return dists[idx_min], idx_min[0], idx_min[1]

def center_of_mass(image, x, y, size=1):
    ys, xs = np.where(image[y-size:y+size+1, x-size:x+size+1])
    if len(ys)==0:
        return None
    center_mass = np.array([xs.mean()+x-size, ys.mean()+y-size])
    return center_mass

def remove_close_nodes(image, points, r=5):
    if len(points)==0:
        return points
    
    def _find_and_remove_one(points):
        d, idx1, idx2 = find_closest(points)
        if idx1!=idx2:
            if d < r:
                p1 = points[idx1]
                p2 = points[idx2]

                m1 = center_of_mass(image, *p1, size=3)
                m2 = center_of_mass(image, *p2, size=3)
                #print(np.any(np.isnan(image)), m1, m2, p1, p2)
                if m1 is None or m2 is None:
                    points = np.delete(points, idx1, axis=0)
                elif euclidean(m1, p1) < euclidean(m2, p2):
                    points = np.delete(points, idx2, axis=0)
                else:
                    points = np.delete(points, idx1, axis=0)                    
                return points
        return None

    while True:
        res = _find_and_remove_one(points)
        if res is None:
            break
        else:
            points = res

    return points

def find_intersections_and_end_of_lines(skeleton, mask_grid=None, nodes_spacing=2):
    
    _skeleton = skeleton.copy()
    _skeleton[ 0, :] = False
    _skeleton[-1, :] = False
    _skeleton[ :, 0] = False
    _skeleton[ :,-1] = False
    
    intersections = []
    end_of_lines = []    
    for y,x in zip(*np.where(_skeleton)):
        patch = _skeleton[neighbour[:,1]+y, neighbour[:,0]+x].tolist()
        
        if patch in valid_intersection:
            intersections.append((x,y))
        elif patch in valid_end_of_lines:
            end_of_lines.append((x,y))   
            
    intersections = np.array(list(set(intersections)))
    end_of_lines = np.array(list(set(end_of_lines)))
        
    if mask_grid is not None:
        intersections_grid = np.vstack(np.where(np.logical_and(_skeleton, mask_grid))).T[:,[1,0]]
        intersections = np.vstack([intersections, intersections_grid])
                      
    intersections = remove_close_nodes(_skeleton, intersections, nodes_spacing)
    end_of_lines = remove_close_nodes(_skeleton, end_of_lines, nodes_spacing)
    
    return intersections, end_of_lines

def find_end_of_lines(skeleton, xs=None, ys=None):
    if xs is None or ys is None:
        ys,xs = np.where(skeleton)    

    image = skeleton.copy()
    image[ 0, :] = False
    image[-1, :] = False
    image[ :, 0] = False
    image[ :,-1] = False
    
    end_of_lines = []
    
    for y,x in zip(ys,xs):
        patch = image[neighbour[:,1]+y, neighbour[:,0]+x].tolist()
        
        if patch in valid_end_of_lines:
            end_of_lines.append((x,y))

    return np.array(end_of_lines)

def is_thin_line(image, xs=None, ys=None, random_trials=10):
    if xs is None or ys is None:
        ys,xs = np.where(image)
    
    rs = np.random.randint(0,len(xs), random_trials)
    for r in rs:
        # 7 is the maximum number of active pixels for a thin line
        # excluding intersections
        if np.sum(image[ys[r]-1:ys[r]+2, xs[r]-1:xs[r]+2])>7:
            return False
    return True

import queue
def find_connectivity(img, x, y):
    
    _img = img.copy()
    
    dy = [0, 0, 1, 1, 1, -1, -1, -1]
    dx = [1, -1, 0, 1, -1, 0, 1, -1]
    xs = []
    ys = []
    q = queue.Queue()
    if _img[y,x] == True:
        q.put((y,x))
    while q.empty() == False:
        v,u = q.get()
        xs.append(u)
        ys.append(v)
        for k in range(8):
            yy = v + dy[k]
            xx = u + dx[k]
            if _img[yy][xx] == True:
                _img[yy][xx] = False
                q.put((yy, xx))
    return xs, ys

def follow_line(img, x, y):
    
    _img = img.copy()
    
    mask = np.array([[True,True,True],[True,False,True],[True,True,True]])
    xs = []
    ys = []

    patch = _img[neighbour[:,1]+y, neighbour[:,0]+x].tolist()
    patch = [x*1 for x in patch]
    if patch not in valid_end_of_lines:
        raise ValueError("starting point is not an end-of-line")

    xs.append(x)
    ys.append(y)  
    _img[y,x] = False
    stop = False
    while not stop:
        stop = True
        dy,dx = np.where(np.logical_and(_img[y-1:y+2, x-1:x+2], mask))
        if len(dx)>0:
            x += dx[0]-1
            y += dy[0]-1
            _img[y][x] = False
            xs.append(x)
            ys.append(y) 
            stop = False
    return xs, ys

def retrieve_graph(skeleton, grid_spacing=-1, nodes_spacing=3, proper_lines=True, junction_size=1):
    
    _skeleton = skeleton.copy()
    
    if grid_spacing is not None and grid_spacing>0:
        
        mask_grid = np.zeros_like(skeleton)
        for y in np.arange(0, skeleton.shape[0], grid_spacing):
            mask_grid[int(y)] = 1
        for x in np.arange(0, skeleton.shape[1], grid_spacing):
            mask_grid[:,int(x)] = 1        
        
        intersections, end_of_lines = find_intersections_and_end_of_lines(skeleton, mask_grid, 
                                                                          nodes_spacing=nodes_spacing) 
    else:
        intersections, end_of_lines = find_intersections_and_end_of_lines(skeleton, nodes_spacing=nodes_spacing)   
    
    
    # disconnect all lines    
    for ix,iy in intersections:
        #_skeleton[iy-1:iy+2, ix-1:ix+2] = 0
        _skeleton[iy-junction_size:iy+junction_size+1, ix-junction_size:ix+junction_size+1] = 0

    # make sure no pixel are active on the borders
    _skeleton[ 0, :] = False
    _skeleton[-1, :] = False
    _skeleton[ :, 0] = False
    _skeleton[ :,-1] = False 

    # nodes are added in the constructor
    i = 0
    graph = nx.Graph()
    for node in intersections:
        graph.add_node(i, pos=node, eol=False) 
        i += 1    
    for node in end_of_lines:
        graph.add_node(i, pos=node, eol=True) 
        i += 1
    nodes_positions = np.vstack((intersections, end_of_lines))

    # find lines and add edges to the graph
    unexpected = []
    ccomponents = measure.label(_skeleton, neighbors=8, background=0)
    for cc in range(ccomponents.max()):
        
        component = ccomponents==cc
        ys,xs = np.where(component)

        # skip lines that are less than 3 pixel long
        if len(ys)<3:
            continue
            
        # check if component is a thin line or a blob
        if is_thin_line(component,xs,ys):
            
            eol = find_end_of_lines(component,xs,ys)            
            if len(eol)!=2:
                # a line can only have 2 end points 
                unexpected.append(component)
            else:
                start_node = np.linalg.norm(eol[0][None]-nodes_positions, axis=1).argmin()
                end_node = np.linalg.norm(eol[1][None]-nodes_positions, axis=1).argmin()
                              
                if proper_lines:
                    # we make sure that the pixels' positions in xs and ys are ordered correctly one another
                    # and following the direction of the line in the image
                    idx_start = np.linalg.norm(eol[0][None]-np.vstack([xs,ys]).T, axis=1).argmin()
                    xs, ys = follow_line(component, xs[idx_start], ys[idx_start])
                
                graph.add_edge(start_node, end_node, line=(xs,ys), idx=cc)
        else:
            # connected component is not a line
            pass

    return graph
'''
def construct_graph(skeleton):
    
    intersections, end_of_lines = find_intersections_and_end_of_lines(skeleton, min_dist=3)   
    
    _skeleton = skeleton.copy()
    
    # disconnect all lines    
    for ix,iy in intersections:
        _skeleton[iy-1:iy+2, ix-1:ix+2] = 0
        #_skeleton[iy-2:iy+3, ix-2:ix+3] = 0

    # make sure no pixel are active on the borders
    _skeleton[ 0, :] = False
    _skeleton[-1, :] = False
    _skeleton[ :, 0] = False
    _skeleton[ :,-1] = False 

    # nodes are added in the constructor
    i = 0
    graph = nx.Graph()
    nodes_positions = np.vstack((intersections, end_of_lines))
    for node in nodes_positions:
        graph.add_node(i, pos=node) 
        i += 1

    # find lines and add edges to the graph
    unexpected = []
    ccomponents = measure.label(_skeleton, neighbors=8, background=0)
    for cc in range(ccomponents.max()):
        component = ccomponents==cc
        ys,xs = np.where(component)
        
        # skip lines that are less than 3 pixel long
        if len(ys)<3:
            continue
            
        # check if component is a thin line or a blob
        if is_thin_line(component,xs,ys):
            eol = find_end_of_lines(component,xs,ys)
            if len(eol)==0 or len(eol)>2:
                # a line can only have 2 end points 
                unexpected.append(component)
            else:
                start_node = np.linalg.norm(eol[0][None]-nodes_positions, axis=1).argmin()
                end_node = np.linalg.norm(eol[1][None]-nodes_positions, axis=1).argmin()
                              
                # we make sure that the pixels' positions in xs and ys are ordered correctly one another
                # and following the direction of the line in the image
                idx_start = np.linalg.norm(eol[0][None]-np.vstack([xs,ys]).T, axis=1).argmin()
                xs, ys = follow_line(component, xs[idx_start], ys[idx_start])
                
                new_nodes = np.vstack([xs,ys]).T
                graph.add_node(i, pos=new_nodes[0])
                graph.add_edge(start_node, i)  
                 
                for pos in new_nodes[1:]:    
                    i += 1
                    graph.add_node(i, pos=pos)
                    graph.add_edge(i-1, i) 
                graph.add_edge(i, end_node)  
                i+=1
        else:
            # connected component is not a line
            pass
    
    return graph
'''
def plot_graph(graph, node_size=200, font_size=6, 
               matched=[], false_pos=[], false_neg=[], 
               node_color=None, edge_color='c', **kwargs):
  
    if node_color is None:
        node_color = []
        for i in graph.nodes():
            if i in matched:
                node_color.append('g')
            elif i in false_pos:
                node_color.append('r')  
            elif i in false_neg:
                node_color.append('b') 
            else:
                node_color.append('y')  
        plt.plot([],[],'y.',label='Unassigned', markersize=15)
        plt.plot([],[],'g.',label='True positive', markersize=15)
        plt.plot([],[],'r.',label='False positive', markersize=15)
        plt.plot([],[],'b.',label='False negative', markersize=15)

    pos = dict({i:graph.nodes.data()[i]['pos'] for i in graph.nodes()})
    nx.draw_networkx(graph, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, **kwargs)
    plt.gca().invert_yaxis()
    plt.legend()
    
def simple_matching(gt_positions, detections, radius_match=0.5):
    
    gt_positions = np.array(gt_positions)
    detections = np.array(detections) 
    
    n_gts = gt_positions.shape[0]    
    n_dets = detections.shape[0]  
    
    matching_distances = []
    matched_gt = np.zeros(len(gt_positions), np.bool)
    matched_det = np.zeros(len(detections), np.bool)    
    TP = []    
    
    for i_gt in range(n_gts): 
        
        for i_d in range(n_dets):

            d = np.sqrt(((detections[i_d,0] - gt_positions[i_gt,0])**2 + \
                         (detections[i_d,1] - gt_positions[i_gt,1])**2))

            if d <= radius_match:
                matching_distances.append(d)
                TP.append((i_gt,i_d))
                matched_gt[i_gt] = True
                matched_det[i_d] = True
                break
            
    TP = np.array(TP)
    FP = np.where(matched_det==False)[0]
    FN = np.where(matched_gt==False)[0]

    return matching_distances, TP[:,0], TP[:,1], FP, FN     

def hungarian_matching(gt_positions, detections, radius_match=0.5):
    
    gt_positions = np.array(gt_positions)
    detections = np.array(detections)   

    n_gts = gt_positions.shape[0]    
    n_dets = detections.shape[0]

    n_max = max(n_gts, n_dets)

    # building the cost matrix based on the distance between 
    # detections and ground-truth positions
    matrix = np.ones((n_max, n_max))*99999
    for i_gt in range(n_gts):    
        for i_d in range(n_dets):

            d = np.sqrt(((detections[i_d,0] - gt_positions[i_gt,0])**2 + \
                         (detections[i_d,1] - gt_positions[i_gt,1])**2))

            if d <= radius_match:
                matrix[i_gt,i_d] = d

    m = Munkres()
    indexes = m.compute(matrix.copy())

    matching_distances = []
    matched_gt = np.zeros(len(gt_positions), np.bool)
    matched_det = np.zeros(len(detections), np.bool)    
    TP_gt = []
    TP_d = []
    for i_gt, i_d in indexes:
        value = matrix[i_gt][i_d]
        if value <= radius_match:
            TP_gt.append(i_gt)
            TP_d.append(i_d)
            matching_distances.append(value)
            matched_gt[i_gt] = True
            matched_det[i_d] = True
            
    TP_gt = np.int_(TP_gt)
    TP_d = np.int_(TP_d)
    FP = np.where(matched_det==False)[0]
    FN = np.where(matched_gt==False)[0]

    return matching_distances, TP_gt, TP_d, FP, FN   

def find_isolated_nodes(graph):
    return list(nx.isolates(graph))

def find_connected_components(graph):
    return nx.connected_components(graph)

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def combine_edge_paths(graph, path):
    xs = []
    ys = []

    _xs, _ys = graph[path[0]][path[1]][0]['line']
    if euclidean(graph.nodes.data()[path[0]]['pos'], np.array([_xs[0], _ys[0]]))<5:

        for i in range(len(path)-1):
            i0 = path[i]
            i1 = path[i+1]
            _xs, _ys = graph[i0][i1][0]['line']
            xs.append(_xs)
            ys.append(_ys)
    else:
        _path = reversed(path)
        for i in range(len(path)-1):
            i0 = path[i]
            i1 = path[i+1]
            _xs, _ys = graph[i0][i1][0]['line']
            if euclidean(graph.nodes.data()[i0]['pos'], np.array([_xs[-1], _ys[-1]]))<5:
                _xs = np.flip(_xs)
                _ys = np.flip(_ys)        
            xs.append(_xs)
            ys.append(_ys)    

    xs = np.concatenate(xs)
    ys = np.concatenate(ys) 
    
    return xs, ys

def edge_distances(graph_gt, graph_pred, s, t, idx_edge_gt, matched_gt, matched_pred):
    
    # given start and end node in ground-truth graph,
    # find corresponding nodes in the prediction, only if they exists!
    if s not in matched_gt or t not in matched_gt:
        return False
    
    # find start and end nodes in the predicted graph
    idx_s_pred = np.where(matched_gt==s)[0][0]
    idx_t_pred = np.where(matched_gt==t)[0][0]
    s_p = matched_pred[idx_s_pred]
    t_p = matched_pred[idx_t_pred]
        
    # ground-truth line
    for edge_gt in graph_gt[s][t].values():
        xs_gt, ys_gt = edge_gt['line']
        if idx_edge_gt == edge_gt['idx']:
            break
    
    # we make sure the first element of the line coincides with the start node
    if euclidean(graph_gt.nodes.data()[s]['pos'], np.array([xs_gt[-1], ys_gt[-1]]))<5:
        xs_gt = np.flip(xs_gt)
        ys_gt = np.flip(ys_gt)    
    
    # find all possible paths in the graph
    paths = list(nx.all_simple_paths(graph_pred, s_p, t_p))
    
    similarities = []
    lines = []
    for path in paths:

        # path in grap_pred may be composed of more than two nodes,
        # therefore we need to combine the paths for each pair of nodes
        xs, ys = combine_edge_paths(graph_pred, path)

        # compute similarity between paths
        distance, path = fastdtw(np.vstack([ys_gt, xs_gt]).T, np.vstack([ys, xs]).T, dist=euclidean)
        distance /= len(xs_gt) 
        
        similarities.append(distance)
        lines.append((xs, ys))
        
    return similarities, paths, lines