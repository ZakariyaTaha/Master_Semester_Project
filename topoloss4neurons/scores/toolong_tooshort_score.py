import numpy as np
import networkx as nx
import queue
import itertools
from scipy.spatial.distance import euclidean

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

def find_connectivity_3d(img, x, y, z, stop=None):
    
    _img = img.copy()   
    _img2 = img.copy()
    
    dx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
    dz = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]     
    xs = []
    ys = []
    zs = []
    cs = []
    q = queue.Queue()
    if _img[y,x,z] == True:
        q.put((x,y,z))
    i = 0
    while q.empty() == False:
        i+=1
        u,v,w = q.get()
        xs.append(u)
        ys.append(v)
        zs.append(w)
        adjacent = [(u,v,w)]
        if stop is not None and i==stop:
            return xs, ys, zs, cs
        for k in range(26):            
            xx = u + dx[k]  
            yy = v + dy[k]
            zz = w + dz[k]            
            if _img[yy, xx, zz] == True:
                _img[yy, xx, zz] = False
                q.put((xx,yy,zz))               
            if _img2[yy, xx, zz] == True:
                adjacent.append((xx,yy,zz))
        cs.append(adjacent)
    return xs, ys, zs, cs
    
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
    
    graph = nx.Graph()

    for cs in css:
        for pos in cs:
            if not graph.has_node(pos):
                graph.add_node(pos, pos=np.array(pos))  
        '''   
        if len(cs)==2:
            us,vs = [cs[0]],[cs[1]]
            distances = [euclidean(cs[0],cs[1])]
        elif len(cs)==3:
            d1 = euclidean(cs[0],cs[1])
            d2 = euclidean(cs[1],cs[2])
            d3 = euclidean(cs[0],cs[2])
            if d1>d2 and d1>d3:
                us,vs = [cs[1],cs[0]],[cs[2],cs[2]] 
                distances = [d2,d3]
            if d2>d1 and d2>d3:
                us,vs = [cs[0],cs[0]],[cs[1],cs[2]]  
                distances = [d1,d3]
            if d3>d1 and d3>d2:
                us,vs = [cs[0],cs[1]],[cs[1],cs[2]]
                distances = [d1,d2]
        else:
        '''
        us,vs = [],[]
        for u,v in itertools.combinations(cs, 2):
            us += [u]
            vs += [v]                    
        distances = [euclidean(u,v) for u,v in zip(us,vs)] 

        for u,v,d in zip(us,vs,distances):
            if not graph.has_edge(u,v):
                '''
                if False:
                    try:
                        length = nx.shortest_path_length(graph,u,v)
                        if length>5:
                            graph.add_edge(u,v)
                    except:
                        graph.add_edge(u,v)      
                else:
                '''
                if d<1.42:
                    graph.add_edge(u,v)

    return graph

def create_graph_3d(skeleton):
    
    _skeleton = skeleton.copy()>0
    
    # make sure no pixel are active on the borders
    _skeleton[ 0, :, :] = False
    _skeleton[-1, :, :] = False
    _skeleton[ :, 0, :] = False
    _skeleton[ :,-1, :] = False   
    _skeleton[ :, :, 0] = False
    _skeleton[ :, :,-1] = False     

    css = []
    while True:
        ys, xs, zs = np.where(_skeleton)
        if len(ys)==0:
            break
        _xs, _ys, _zs, _cs = find_connectivity_3d(_skeleton, xs[0], ys[0], zs[0], stop=None)
        css += _cs
        _skeleton[_ys,_xs,_zs] = False       
    
    graph = nx.Graph()

    for cs in css:
        for pos in cs:
            if not graph.has_node(pos):
                graph.add_node(pos, pos=np.array(pos))  

        us,vs = [],[]
        for u,v in itertools.combinations(cs, 2):
            us += [u]
            vs += [v]                    
        distances = [euclidean(u,v) for u,v in zip(us,vs)] 

        for u,v,d in zip(us,vs,distances):
            if not graph.has_edge(u,v):
                if d<2.1:
                    graph.add_edge(u,v)

    return graph

def extract_gt_paths(graph_gt, N=100, min_path_length=10):

    cc_graphs = list(graph_gt.subgraph(c) for c in nx.connected_components(graph_gt))
    n_subgraph = len(cc_graphs)  
    
    total = 0

    paths = []
    for _ in range(N*1000):
        
        idx_sub = np.random.choice(np.arange(n_subgraph), 1)[0]
        graph = cc_graphs[idx_sub]
        
        nodes_gt = list(graph.nodes()) 
        n_nodes = len(nodes_gt)
        if n_nodes < 2:
            continue
    
        # randomly pick two node in the GT
        idx_s,idx_t = np.random.choice(np.arange(n_nodes), 2, replace=False)
        s_gt, t_gt = nodes_gt[idx_s], nodes_gt[idx_t]
        
        # search shortest path in GT
        try:
            shortest_path_gt = list(nx.shortest_path(graph, tuple(s_gt), tuple(t_gt)))
            #shortest_path_gt = list(nx.astar_path(graph, tuple(s_gt), tuple(t_gt)))
            length_line_gt = len(shortest_path_gt)
        except:
            # path not found
            continue 
            
        if length_line_gt<min_path_length:
            continue
            
        paths.append({"s_gt":s_gt, "t_gt":t_gt, "shortest_path_gt":shortest_path_gt})
        
        total += 1
        
        if total==N:
            break

    return paths
    
def toolong_tooshort_score(paths_gt, graph_pred, radius_match=5, length_deviation=0.05):
    """
    A higher-order CRF model for road network extraction
    Jan D. Wegner, Javier A. Montoya-Zegarra, Konrad Schindler
    2013
    
    These are
    computed in the following way: we randomly sample two
    points which lie both on the true and the estimated road
    network, and check whether the shortest path between the
    two points has the same length in both networks (up to a
    deviation of 5% to account for geometric uncertainty). We
    then keep repeating this procedure with different random
    points and record the percentages of correct, too short, too
    long and infeasible paths, until these percentages have converged. 
    Infeasible and too long paths indicate missing links,
    whereas too short ones indicate hallucinated connections.  
    
    """
      
    nodes_pred = np.array(graph_pred.nodes())
    idxs_pred = np.arange(len(nodes_pred))     

    counter_correct = 0
    counter_toolong = 0
    counter_tooshort = 0
    counter_infeasible = 0
    
    res = []  
    for path in paths_gt:
    
        # unpack GT path
        s_gt, t_gt = np.array(path["s_gt"]), np.array(path["t_gt"])
        shortest_path_gt = path["shortest_path_gt"]
        length_line_gt = len(shortest_path_gt)

        # match GT nodes in prediction
        nodes_radius_s = nodes_pred[np.linalg.norm(nodes_pred-s_gt[None], axis=1)<radius_match]
        nodes_radius_t = nodes_pred[np.linalg.norm(nodes_pred-t_gt[None], axis=1)<radius_match]
        if len(nodes_radius_s)==0 or len(nodes_radius_t)==0:
            counter_infeasible += 1
            res.append({"line_gt":shortest_path_gt, 
                        "line_pred":None,
                        "s_gt":s_gt,"t_gt":t_gt,
                        "s_pred":None, "t_pred":None,
                        "tooshort":False, "toolong":False,
                        "correct":False, "infeasible":True})
            continue
        s_pred = nodes_radius_s[np.linalg.norm(nodes_radius_s-s_gt[None], axis=1).argmin()]
        t_pred = nodes_radius_t[np.linalg.norm(nodes_radius_t-t_gt[None], axis=1).argmin()]
        
        # find shortest path in prediction
        try:
            shortest_path_pred = list(nx.shortest_path(graph_pred, tuple(s_pred), tuple(t_pred)))
            #shortest_path_pred = list(nx.astar_path(graph_pred, tuple(s_pred), tuple(t_pred)))
            length_line_pred = len(shortest_path_pred)
        except:
            # path not found
            counter_infeasible += 1
            res.append({"line_gt":shortest_path_gt, 
                        "line_pred":None,
                        "s_gt":s_gt,"t_gt":t_gt,
                        "s_pred":s_pred, "t_pred":t_pred,
                        "tooshort":False, "toolong":False,
                        "correct":False, "infeasible":True})            
            continue 

        # compare path lengths
        toolong, tooshort, correct = False,False,False        
        if length_line_pred>length_line_gt*(1+length_deviation):
            toolong = True
            counter_toolong += 1
        elif length_line_pred<length_line_gt*(1-length_deviation): 
            tooshort = True
            counter_tooshort += 1                   
        else:
            correct = True
            counter_correct += 1
            
        res.append({"line_gt":shortest_path_gt, 
                    "line_pred":shortest_path_pred,
                    "s_gt":s_gt,"t_gt":t_gt,
                    "s_pred":s_pred, "t_pred":t_pred,
                    "tooshort":tooshort, "toolong":toolong,
                    "correct":correct, "infeasible":False}) 
            
    total = len(paths_gt)
        
    return total, counter_correct/total, counter_tooshort/total, counter_toolong/total, counter_infeasible/total, res    