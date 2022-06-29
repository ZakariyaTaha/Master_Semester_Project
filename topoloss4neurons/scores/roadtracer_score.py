import numpy as np
from scipy.spatial.distance import euclidean

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
                      
    #intersections = remove_close_nodes(_skeleton, intersections, nodes_spacing)
    #end_of_lines = remove_close_nodes(_skeleton, end_of_lines, nodes_spacing)
    
    return intersections, end_of_lines

def clustering(intersections, d=10):
    _intersections = intersections.copy()

    clusters = []
    found = True
    while found:
        found = False
        for point in _intersections:
            idx_cluster = np.where(np.linalg.norm(_intersections-point[None], axis=1)<d)[0]
            if len(idx_cluster)>1:
                cluster_center = _intersections[idx_cluster].mean(0)
                clusters.append(cluster_center)
                _intersections = _intersections[np.setdiff1d(range(len(_intersections)),idx_cluster)]
                found = True
                break

    clusters = np.array(clusters)
    return clusters

def edge_directions(image, x,y, r=10):

    x,y = int(x),int(y)
    patch = image[y-r:y+r+1, x-r:x+r+1].copy()
    patch[1:-1, 1:-1] = 0
    
    directions = []    
    for y,x in zip(*np.where(patch)):
        
        vector = np.array([x,y])-np.array(patch.shape)/2
        vector = vector / np.linalg.norm(vector)
        
        directions.append(vector)
        
    return np.array(directions)

def compute_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_score_node(image_gt, image_pred, p_gt, p_pred, radius_directions=10):
    
    directions_gt = edge_directions(image_gt, *p_gt, radius_directions)
    directions_pred = edge_directions(image_pred, *p_pred, radius_directions)
    if len(directions_gt)==0 or len(directions_pred)==0:
        return 0, 1, None

    matched_gt = np.zeros(len(directions_gt), np.bool)
    for i,d_gt in enumerate(directions_gt):

        for d_pred in directions_pred:
            if compute_angle(d_gt, d_pred) < np.pi/2:
                matched_gt[i] = True
                break

    matched_pred = np.zeros(len(directions_pred), np.bool)
    for i,d_pred in enumerate(directions_pred):

        for d_gt in directions_gt:
            if compute_angle(d_pred, d_gt) < np.pi/2:
                matched_pred[i] = True
                break
    
    f_v_correct = matched_gt.sum()/len(matched_gt)
    f_u_error = (matched_pred==False).sum()/len(matched_pred)
    
    debug = {"pos_gt": p_gt, "pos_pred":p_pred, 
             "f_v_correct":f_v_correct, 
             "f_u_error":f_u_error,
             "directions_gt":directions_gt,
             "directions_pred":directions_gt}
    
    return f_v_correct, f_u_error, debug

def roadtracer_score(label_s, pred_mask_s, radius_match=60, radius_directions=10, clustering_d=10):

    intersections_gt, _ = find_intersections_and_end_of_lines(label_s, nodes_spacing=0)
    intersections_pred, _ = find_intersections_and_end_of_lines(pred_mask_s, nodes_spacing=0)
    
    intersections_gt = clustering(intersections_gt, d=clustering_d)
    intersections_pred = clustering(intersections_pred, d=clustering_d)
    
    n_correct = 0
    n_error = 0
    
    matched_gt = np.zeros(len(intersections_gt), np.bool)
    matched_pred = np.zeros(len(intersections_pred), np.bool)
    debugs = []
    
    for i_gt, node_pos_gt in enumerate(intersections_gt):
        
        match = -1
        bestCorrectScore = -1
        bestExtraScore = -1
        bestFscore = -1
        bestDebug = None
        for i_pred, node_pos_pred in enumerate(intersections_pred):

            if matched_pred[i_pred]==False:
                if euclidean(node_pos_gt, node_pos_pred)<radius_match:

                    correctScore, extraScore, debug = compute_score_node(label_s, pred_mask_s, 
                                                                         node_pos_gt, node_pos_pred,
                                                                          radius_directions)
                    
                    # bullshit copied from original road tracer code
                    score = correctScore - extraScore
                    if score <= 0:
                        continue

                    # bullshit copied from original road tracer code
                    fscore = score - euclidean(node_pos_gt, node_pos_pred) / radius_match
                    if match==-1 or fscore > bestFscore:
                        match = i_pred
                        bestCorrectScore = correctScore
                        bestExtraScore = extraScore
                        bestFscore = fscore
                        bestDebug = debug
                    
        if match!=-1:
            n_correct += bestCorrectScore
            n_error += bestExtraScore             
            matched_pred[match] = True
            matched_gt[i_gt] = True
            
            bestDebug["matched"]=True           
            debugs.append(bestDebug)

    for i,matched in enumerate(matched_pred):
        if matched==False:
            debug = {"matched":False,
                     "pos_pred":intersections_pred[i]}          
            debugs.append(debug)
            
    n_error += np.sum(matched_pred==False)

    n_junctions_gt = len(intersections_gt) 
    if n_junctions_gt==0:
        print("no juntion detected in ground_truth!")
        return 0,1,[]
        
    F_correct = n_correct/n_junctions_gt
    F_error = n_error/(n_error+n_correct)
    
    return F_correct, F_error, debugs