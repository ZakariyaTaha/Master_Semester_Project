import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import imageio
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import networkx as nx
import itertools
import queue
from skimage.morphology import skeletonize, binary_dilation

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
'''
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
'''
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

def retrieve_graph2(skeleton, grid_spacing=-1, proper_lines=True, junction_size=1):
    
    _skeleton = skeleton.copy()
    
    if grid_spacing is not None and grid_spacing>0:
        
        mask_grid = np.zeros_like(skeleton)
        for y in np.arange(0, skeleton.shape[0], grid_spacing):
            mask_grid[int(y)] = 1
        for x in np.arange(0, skeleton.shape[1], grid_spacing):
            mask_grid[:,int(x)] = 1        
        
        intersections, end_of_lines = find_intersections_and_end_of_lines(skeleton, mask_grid, 
                                                                          nodes_spacing=0) 
    else:
        intersections, end_of_lines = find_intersections_and_end_of_lines(skeleton, nodes_spacing=0)   
    
    
    # disconnect all lines    
    img_intersections = np.zeros_like(_skeleton)  
    #print(img_intersections.shape, skeleton.shape, _skeleton.shape)
    for ix,iy in intersections:
        patch = _skeleton[iy-junction_size:iy+junction_size+1, ix-junction_size:ix+junction_size+1]    
        img_intersections[iy-junction_size:iy+junction_size+1, ix-junction_size:ix+junction_size+1] += patch
        _skeleton[iy-junction_size:iy+junction_size+1, ix-junction_size:ix+junction_size+1] = 0
        
    intersections = []
    ccomponents = measure.label(img_intersections>0, neighbors=8, background=0)
    for cc in range(1, ccomponents.max()):
        component = ccomponents==cc
        ys,xs = np.where(component)
        intersections.append((np.mean(xs), np.mean(ys)))
    intersections = np.array(intersections)        

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
    for cc in range(1,ccomponents.max()):
        
        component = ccomponents==cc
        ys,xs = np.where(component)

        # skip lines that are less than 3 pixel long
        #if len(ys)<3:
        #    continue
            
        # check if component is a thin line or a blob
        #if is_thin_line(component,xs,ys):
            
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
        #else:
        # connected component is not a line
        #pass

    return graph#, img_intersections, intersections, intersections2

def get_nodes(graph, intersections_only=True):
    node_idxs = []
    node_poss = []
    for idx, data in graph.nodes.data():
        if data['eol']==False:
            # intersection
            node_idxs.append(idx)
            node_poss.append(data['pos'].tolist())        
        else:
            # end of line
            if not intersections_only:
                node_idxs.append(idx)
                node_poss.append(data['pos'].tolist())
    node_idxs = np.array(node_idxs)         
    node_poss = np.array(node_poss) 
    return node_idxs, node_poss

def interruptions_score(label_s, pred_mask_s, 
                       nodes_spacing=3, junction_size=2,
                       radius_match=15, th_similarity=0.35,
                       intersections_only=True):
    
    #graph_gt = retrieve_graph(label_s, nodes_spacing=nodes_spacing, junction_size=junction_size)
    #graph_pred = retrieve_graph(pred_mask_s, nodes_spacing=nodes_spacing, junction_size=junction_size)
    graph_gt = retrieve_graph2(label_s, junction_size=1)
    graph_pred = retrieve_graph2(pred_mask_s, junction_size=1)    
    
    node_idxs_gt, node_poss_gt = get_nodes(graph_gt, intersections_only=intersections_only)
    node_idxs_pred, node_poss_pred = get_nodes(graph_pred, intersections_only=intersections_only)

    res = []
    for s_gt, t_gt, data in graph_gt.edges.data():

        if intersections_only:
            # consider intersections only, no end of lines
            if graph_gt.node.data()[s_gt]["eol"] or graph_gt.node.data()[t_gt]["eol"]:
                continue

        xs_gt,ys_gt = data['line']
        length_line_gt = len(xs_gt)

        s_pos_gt = node_poss_gt[node_idxs_gt[s_gt]]
        t_pos_gt = node_poss_gt[node_idxs_gt[t_gt]]

        candidates_s_pred = node_idxs_pred[np.linalg.norm(node_poss_pred-s_pos_gt[None], axis=1)<radius_match]
        candidates_t_pred = node_idxs_pred[np.linalg.norm(node_poss_pred-t_pos_gt[None], axis=1)<radius_match]

        best_similarity = np.inf
        best_xs = None
        best_ys = None 
        connected = False
        best_s_pred = None
        best_t_pred = None
        for s_pred in candidates_s_pred:
            for t_pred in candidates_t_pred:
                try:
                    shortest_path_pred = list(nx.shortest_path(graph_pred, s_pred, t_pred))
                except:
                    # path not found
                    continue 

                # compose line
                xs = []
                ys = []
                for i in range(len(shortest_path_pred)-1):
                    i0 = shortest_path_pred[i]
                    i1 = shortest_path_pred[i+1]
                    _xs, _ys = graph_pred[i0][i1]['line']
                    xs += _xs
                    ys += _ys
                length_line_pred = len(xs)

                similarity = np.abs(length_line_gt-length_line_pred)/(length_line_gt)
                if similarity<th_similarity and similarity<best_similarity:
                    best_similarity = similarity
                    best_xs = xs
                    best_ys = ys
                    best_s_pred = s_pred
                    best_t_pred = t_pred
                    connected = True

        res.append({"line_gt":np.vstack([xs_gt,ys_gt]).T, 
                    "line_pred":np.vstack([best_xs,best_ys]).T,
                    "best_similarity":best_similarity,
                    "connected":connected,
                    "s_gt":s_gt,"t_gt":t_gt,
                    "s_pred":best_s_pred, "t_pred":best_t_pred})   

    connected = np.array([x["connected"] for x in res])
    p_connected = connected.sum()/len(connected)

    return p_connected, res

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
    
def find_connectivity(img, x, y, stop=None):
    
    _img = img.copy()
    _img2 = img.copy()
    
    dy = [0, 0, 1, 1, 1, -1, -1, -1]
    dx = [1, -1, 0, 1, -1, 0, 1, -1]
    xs = []
    ys = []
    cs = []
    q = queue.Queue()
    if _img[y,x] == True:
        q.put((y,x))
    i = 0
    while q.empty() == False:
        i+=1
        v,u = q.get()
        xs.append(u)
        ys.append(v)
        adjacent = [(u,v)]
        if stop is not None and i==stop:
            return xs, ys, cs
        for k in range(8):
            yy = v + dy[k]
            xx = u + dx[k]            
            if _img[yy, xx] == True:
                _img[yy, xx] = False
                q.put((yy, xx))               
            if _img2[yy, xx] == True:
                adjacent.append((xx,yy))
        cs.append(adjacent)
    return xs, ys, cs    
    
def create_graph(skeleton):
    
    _skeleton = skeleton.copy()>0

    # make sure no pixel are active on the borders
    _skeleton[ 0, :] = False
    _skeleton[-1, :] = False
    _skeleton[ :, 0] = False
    _skeleton[ :,-1] = False

    css = []
    while True:
        ys, xs = np.where(_skeleton)
        if len(ys)==0:
            break
        _xs, _ys, _cs = find_connectivity(_skeleton, xs[0], ys[0], stop=None)
        css += _cs
        _skeleton[_ys,_xs] = False       
    
    graph = nx.MultiGraph()

    for cs in css:
        for pos in cs:
            if not graph.has_node(pos):
                graph.add_node(pos, pos=np.array(pos))  


        us,vs = [],[]
        for u,v in itertools.combinations(cs, 2):
            us += [u]
            vs += [v]
        distances = [euclidean(u,v) for u,v in zip(us,vs)]
        idxs = np.argsort(distances)   

        us = [us[i] for i in idxs]
        vs = [vs[i] for i in idxs]            

        for u,v in zip(us,vs):
            if not graph.has_edge(u,v):
                try:
                    length = nx.shortest_path_length(graph,u,v)
                    if length>5:
                        graph.add_edge(u,v)
                except:
                    graph.add_edge(u,v)
                    
    return graph