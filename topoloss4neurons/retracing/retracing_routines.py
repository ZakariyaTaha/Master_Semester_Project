import numpy as np
import math
import sys
import scipy.sparse.csgraph as g
from random import randint
from enum import IntEnum
import copy
import time
import graph_tool
import graph_tool.topology
import graph_tool.util

class nodeType(IntEnum):
  NotVisited=0
  Visited=1 #the path from the root to the current node passes through this node
  Processed=2 #the node has been processed, but is not on the current path

class edgeType(IntEnum):
  NoEdge=0
  NotTraversed=1
  OnPath=2
  OffPath=3

class traceType(IntEnum):
  Loop=1
  Path=2

def randomPathOrLoop(edges,nstart):
  # node state; can be: NotVisited,Visited (on path), Processed (off path)
  visited=np.zeros(edges.shape[0]).astype(np.uint) # node state
  traversed=np.zeros_like(edges).astype(np.uint8)
  traversed[edges==1]=edgeType.NotTraversed # edges not traversed yet
  visited[nstart]=nodeType.Visited
  currentPath=[nstart]
  loop=None
  loopEdges=None
  while len(currentPath)>0:
    c=currentPath[-1]
    unseenchildren=np.nonzero(traversed[c]==edgeType.NotTraversed)[0]
    if unseenchildren.shape[0]==0:
      # no children to descend to
      if edges[c].sum()==1 and np.max(traversed[c])==edgeType.OnPath:
        # node with one arc and one that is on path - not after backtracking
        # found a path!
        return currentPath, traversed==edgeType.OnPath, traceType.Path
      # explored all children of that node, backtrack
      visited[c]=nodeType.Processed
      currentPath.pop()
      if len(currentPath)>0:
        traversed[currentPath[-1],c]=edgeType.OffPath
        traversed[c,currentPath[-1]]=edgeType.OffPath
      continue
    # select a random child
    ci=randint(0,unseenchildren.shape[0]-1)
    # descend along an edge and mark it as traversed
    cand=unseenchildren[ci]
    traversed[c,cand]=edgeType.OnPath
    traversed[cand,c]=edgeType.OnPath
    if visited[cand] == nodeType.Visited :
      # found a loop, remember it
      if not loop:
        loop=copy.deepcopy(currentPath)
        loopEdges=traversed==edgeType.OnPath
      # explore other children of c
      traversed[c,cand]=edgeType.OffPath
      traversed[cand,c]=edgeType.OffPath
      continue
    elif visited[cand] == nodeType.Processed :
      # intersection with previously searched path,
      # explore other children of c
      traversed[c,cand]=edgeType.OffPath
      traversed[cand,c]=edgeType.OffPath
      continue
    elif visited[cand] == nodeType.NotVisited:
      # advance depth-first
      currentPath.append(cand)
      visited[cand] = nodeType.Visited

  # didnt find an open path, return a loop  
  return loop, loopEdges, traceType.Loop

def precomputeGraphElements(rad,pred,alpha):

    cost_shape=np.array(pred.shape)
    cost_flat =-alpha*np.log(pred.flatten()) #+sys.float_info.epsilon)

    diam=2*rad+1

    # prepare coordinates of the nodes corresponding to a single node in the path
    #t0=time.time()
    offset_X=np.array(range(-rad,rad+1)).reshape(diam,1,1)
    offset_Y=np.swapaxes(np.copy(offset_X),0,1)
    offset_Z=np.swapaxes(np.copy(offset_X),0,2)
    ox=np.broadcast_to(offset_X,(diam,diam,diam))
    oy=np.broadcast_to(offset_Y,(diam,diam,diam))
    oz=np.broadcast_to(offset_Z,(diam,diam,diam))
    offset=np.stack([ox,oy,oz],axis=-1)
    noff=diam**3
    offset=offset.reshape(noff,3)
    dist2path=(offset**2).sum(1)
    #t1=time.time()
    
    # edges for the subgraph corresponding to a single node in a path
    # can be precomputed once for all path nodes
    # all neighboring nodes are connected in both directions
    off1=offset.reshape(1,noff,3)
    off2=offset.reshape(noff,1,3)
    doff=off1-off2
    doff2=(doff**2).sum(2)
    neighb_intra_mask=np.logical_and(doff2<=3+sys.float_info.epsilon,doff2>0)
    neighb_intra_coords=np.stack(np.nonzero(neighb_intra_mask),1)

    # edges between nodes corresponding to different path nodes
    # depend on the relative position of the path nodes
    # but can also be prepared in advance for the 26 possible cases
    # of relative positions of the previous and current path nodes
    neighb_inter_coords=dict()
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                pdiff=np.array([i,j,k])
                odiff=doff+pdiff
                odiff2=(odiff**2).sum(2)
                dotpr=(pdiff*odiff).sum(2)
                neighb_inter_mask=np.logical_and(odiff2<=3+sys.float_info.epsilon,dotpr>=0)
                neighb_inter_coords[tuple(pdiff)]=np.stack(np.nonzero(neighb_inter_mask),1)+np.array([[0,noff]])
    #print(neighb_inter_coords[(0,1,0)])

    return dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape

def precomputeGraphElements2D(rad,pred,alpha):

    cost_shape=np.array(pred.shape)
    cost_flat =-alpha*np.log(pred.flatten()) #+sys.float_info.epsilon)

    diam=2*rad+1

    # prepare coordinates of the nodes corresponding to a single node in the path
    #t0=time.time()
    offset_X=np.array(range(-rad,rad+1)).reshape(diam,1)
    offset_Y=np.swapaxes(np.copy(offset_X),0,1)
    ox=np.broadcast_to(offset_X,(diam,diam))
    oy=np.broadcast_to(offset_Y,(diam,diam))
    offset=np.stack([ox,oy],axis=-1)
    noff=diam**2
    offset=offset.reshape(noff,2)
    dist2path=(offset**2).sum(1)
    #t1=time.time()
    
    # edges for the subgraph corresponding to a single node in a path
    # can be precomputed once for all path nodes
    # all neighboring nodes are connected in both directions
    off1=offset.reshape(1,noff,2)
    off2=offset.reshape(noff,1,2)
    doff=off1-off2
    doff2=(doff**2).sum(2)
    neighb_intra_mask=np.logical_and(doff2<=2+sys.float_info.epsilon,doff2>0)
    neighb_intra_coords=np.stack(np.nonzero(neighb_intra_mask),1)

    # edges between nodes corresponding to different path nodes
    # depend on the relative position of the path nodes
    # but can also be prepared in advance for the 26 possible cases
    # of relative positions of the previous and current path nodes
    neighb_inter_coords=dict()
    for i in range(-1,2):
        for j in range(-1,2):
            pdiff=np.array([i,j])
            odiff=doff+pdiff
            odiff2=(odiff**2).sum(2)
            dotpr=(pdiff*odiff).sum(2)
            neighb_inter_mask=np.logical_and(odiff2<=2+sys.float_info.epsilon,dotpr>=0)
            neighb_inter_coords[tuple(pdiff)]=np.stack(np.nonzero(neighb_inter_mask),1)+np.array([[0,noff]])
    #print(neighb_inter_coords[(0,1,0)])

    return dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape

# todo: connections to other traces
def graphForRetracing(path_coords,rad,pred,alpha,
     dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape):
    # this function generates a graph used to re-trace a single path
    # through the ground-truth graph
    # the re-tracing, formulated as a problem of sequence alignment, 
    # is itself done by finding a shortest path in a graph;
    # this graph has many nodes corresponding to each node of the path
    # each node corresponds to a pixel/voxel in the input data
    # a cost is associated to traversing a node
    # this cost depends on the distance to the path node
    # and on the strength of the prediction in the corresp voxel
    # the graph_tool library is used to perform the shortest path search
    # this function prepares the graph for the shortest path search
    #
    # i try to add nodes and edges in batches here - 
    # - it speeds up the whole process several times
    # a reference simple implementation of the same procedure can be found below
    # 
    # path_coords - nX3 table of coordinates of path points
    # rad - a scalar, how much the new path can deviate from the old one
    # pred - a prediction volume - re-tracing cost will be calculated from it
    # alpha - the re-tracing cost is a combination of a distance to the corresp
    #         path point and alpha times -log prediction
    
    
    npath=len(path_coords)

    # for each edge we need a cost
    edge_costs=[]
    # for each node - its coordinates in the image
    node_coords=[]
    # and the path node to which it corresponds
    path_corr=[]
    gr=graph_tool.Graph()

    # the source node
    source=gr.add_vertex() 
    node_coords.append(np.zeros((1,3)))
    path_corr.append(np.array([-1]))

    # nodes corresponding to the first path node
    gr.add_vertex(n=noff) 
    # edges from the source node to the nodes corresp to the first path node
    gr.add_edge_list(np.stack((np.zeros(noff),np.array(range(noff))+1),axis=-1)) 
    node_locations=offset+path_coords[0].reshape(1,3)
    node_coords.append(node_locations)
    path_corr.append(np.zeros(noff))
    flattened_inds=np.ravel_multi_index((node_locations[:,0],
                                         node_locations[:,1],
                                         node_locations[:,2]),dims=pred.shape,mode='clip')
    edge_costs.append(cost_flat[flattened_inds]+dist2path)
    # edges between nodes corresponding to the same path node
    coords=neighb_intra_coords+1
    gr.add_edge_list(coords)
    coords=neighb_intra_coords
    flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],
                                         node_locations[coords[:,1],1],
                                         node_locations[coords[:,1],2]),dims=pred.shape,mode='clip')
    edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])

    # nodes corresponding to the consecutive path nodes
    for p in range(1,npath):
        # edges between nodes corresponding to the same path node
        coords=neighb_intra_coords+p*noff+1
        gr.add_edge_list(coords)
        node_locations=offset+path_coords[p].reshape(1,3)
        node_coords.append(node_locations)
        path_corr.append(np.ones(noff)*p)
        coords=neighb_intra_coords
        flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],node_locations[coords[:,1],1],node_locations[coords[:,1],2]),dims=pred.shape,mode='clip')
        edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])
        # edges between nodes corresponding to different path nodes
        pdiff=path_coords[p]-path_coords[p-1]
        coords=neighb_inter_coords[tuple(pdiff)]+(p-1)*noff+1
        gr.add_edge_list(coords)
        node_locations=offset+path_coords[p].reshape(1,3)
        coords=neighb_inter_coords[tuple(pdiff)]-noff # the second column will be used below
        flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],node_locations[coords[:,1],1],node_locations[coords[:,1],2]),dims=pred.shape,mode='clip')
        edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])
        #print(gr)

    target=gr.add_vertex()
    node_coords.append(np.zeros((1,3)))
    path_corr.append(np.array([-1]))
    gr.add_edge_list(np.stack((np.array(range(noff))+(npath-1)*noff+1,np.ones(noff)*gr.vertex_index[target]),axis=-1))
    #print(gr)
    edge_costs.append(np.zeros((noff)))
    edge_cost_v=np.concatenate(edge_costs,axis=0)
    node_coord_a=np.concatenate(node_coords,axis=0)
    edge_cost_map=gr.new_edge_property('double',edge_cost_v)
    node_coord_map=gr.new_vertex_property('vector<int>')
    node_coord_map.set_2d_array(node_coord_a.transpose())
    corr_path_node_map=gr.new_vertex_property('int',np.concatenate(path_corr,axis=0))
    is_target=gr.new_vertex_property('bool',val=False)
    is_target[target]=True
    
    invalid_nodes=np.logical_or(np.any(node_coord_a<0,axis=1),np.any(node_coord_a>=cost_shape.reshape(1,3),axis=1))
    invalid_node_inds=np.nonzero(invalid_nodes)[0]
    # it has to be done like that for technical reasons-
    # sorting and reversing prevents invalidating indexes
    for ini in reversed(sorted(invalid_node_inds)):
        gr.remove_vertex(ini,fast=True)
    #gr.remove_vertex(invalid_node_inds,fast=True) # doesnt work - a bug!
    
    source_ind=0
    target_ind=np.nonzero(is_target.a)[0].item()

    return gr, edge_cost_map,node_coord_map,corr_path_node_map,source_ind,target_ind

# todo: connections to other traces
def graphForRetracing2D(path_coords,rad,pred,alpha,
     dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape):
    # this function generates a graph used to re-trace a single path
    # through the ground-truth graph
    # the re-tracing, formulated as a problem of sequence alignment, 
    # is itself done by finding a shortest path in a graph;
    # this graph has many nodes corresponding to each node of the path
    # each node corresponds to a pixel/voxel in the input data
    # a cost is associated to traversing a node
    # this cost depends on the distance to the path node
    # and on the strength of the prediction in the corresp voxel
    # the graph_tool library is used to perform the shortest path search
    # this function prepares the graph for the shortest path search
    #
    # i try to add nodes and edges in batches here - 
    # - it speeds up the whole process several times
    # a reference simple implementation of the same procedure can be found below
    # 
    # path_coords - nX2 table of coordinates of path points
    # rad - a scalar, how much the new path can deviate from the old one
    # pred - a prediction volume - re-tracing cost will be calculated from it
    # alpha - the re-tracing cost is a combination of a distance to the corresp
    #         path point and alpha times -log prediction
    
    
    npath=len(path_coords)

    # for each edge we need a cost
    edge_costs=[]
    # for each node - its coordinates in the image
    node_coords=[]
    # and the path node to which it corresponds
    path_corr=[]
    gr=graph_tool.Graph()

    # the source node
    source=gr.add_vertex() 
    node_coords.append(np.zeros((1,2)))
    path_corr.append(np.array([-1]))

    # nodes corresponding to the first path node
    gr.add_vertex(n=noff) 
    # edges from the source node to the nodes corresp to the first path node
    gr.add_edge_list(np.stack((np.zeros(noff),np.array(range(noff))+1),axis=-1)) 
    node_locations=offset+path_coords[0].reshape(1,2)
    node_coords.append(node_locations)
    path_corr.append(np.zeros(noff))
    flattened_inds=np.ravel_multi_index((node_locations[:,0],
                                         node_locations[:,1]),dims=pred.shape,mode='clip')
    edge_costs.append(cost_flat[flattened_inds]+dist2path)
    # edges between nodes corresponding to the same path node
    coords=neighb_intra_coords+1
    gr.add_edge_list(coords)
    coords=neighb_intra_coords
    flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],
                                         node_locations[coords[:,1],1]),dims=pred.shape,mode='clip'),
    edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])

    # nodes corresponding to the consecutive path nodes
    for p in range(1,npath):
        # edges between nodes corresponding to the same path node
        coords=neighb_intra_coords+p*noff+1
        gr.add_edge_list(coords)
        node_locations=offset+path_coords[p].reshape(1,2)
        node_coords.append(node_locations)
        path_corr.append(np.ones(noff)*p)
        coords=neighb_intra_coords
        flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],node_locations[coords[:,1],1]),dims=pred.shape,mode='clip')
        edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])
        # edges between nodes corresponding to different path nodes
        pdiff=path_coords[p]-path_coords[p-1]
        coords=neighb_inter_coords[tuple(pdiff)]+(p-1)*noff+1
        gr.add_edge_list(coords)
        node_locations=offset+path_coords[p].reshape(1,2)
        coords=neighb_inter_coords[tuple(pdiff)]-noff # the second column will be used below
        flattened_inds=np.ravel_multi_index((node_locations[coords[:,1],0],node_locations[coords[:,1],1]),dims=pred.shape,mode='clip')
        edge_costs.append(cost_flat[flattened_inds]+dist2path[coords[:,1]])
        #print(gr)

    target=gr.add_vertex()
    node_coords.append(np.zeros((1,2)))
    path_corr.append(np.array([-1]))
    gr.add_edge_list(np.stack((np.array(range(noff))+(npath-1)*noff+1,np.ones(noff)*gr.vertex_index[target]),axis=-1))
    #print(gr)
    edge_costs.append(np.zeros((noff)))
    edge_cost_v=np.concatenate(edge_costs,axis=0)
    node_coord_a=np.concatenate(node_coords,axis=0)
    edge_cost_map=gr.new_edge_property('double',edge_cost_v)
    node_coord_map=gr.new_vertex_property('vector<int>')
    node_coord_map.set_2d_array(node_coord_a.transpose())
    corr_path_node_map=gr.new_vertex_property('int',np.concatenate(path_corr,axis=0))
    is_target=gr.new_vertex_property('bool',val=False)
    is_target[target]=True
    
    invalid_nodes=np.logical_or(np.any(node_coord_a<0,axis=1),np.any(node_coord_a>=cost_shape.reshape(1,2),axis=1))
    invalid_node_inds=np.nonzero(invalid_nodes)[0]
    # it has to be done like that for technical reasons-
    # sorting and reversing prevents invalidating indexes
    for ini in reversed(sorted(invalid_node_inds)):
        gr.remove_vertex(ini,fast=True)
    #gr.remove_vertex(invalid_node_inds,fast=True) # doesnt work - a bug!
    
    source_ind=0
    target_ind=np.nonzero(is_target.a)[0].item()

    return gr, edge_cost_map,node_coord_map,corr_path_node_map,source_ind,target_ind

def renderNodes(lbl_,coords,val):
    for k in range(coords.shape[0]):
        lbl_[coords[k][0],coords[k][1],coords[k][2]]=val
    return lbl_

def renderNodes2D(lbl_,coords,val):
    for k in range(coords.shape[0]):
        lbl_[coords[k][0],coords[k][1]]=val
    return lbl_

def resamplePaths(lbl,predictions,edges,node_coords,rad,alpha):
    # predictions must be non-negative!
    # paint the modified ground truth into lbl
    # by re-tracing the ground truth graph
    # TODO: 
    # 1) take care of connections (ignored here) between different paths
    # 2) take care of proper loop re-tracing
    
    # pre-compute whatever can be taken outside of the loop
    dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape =\
        precomputeGraphElements(rad,predictions,alpha)

    edges_nt=np.copy(edges)
    percentage_sum = edges_nt.sum()
    start_time = time.time()
    while np.any(edges_nt):
        
        print("Percentage-done={:0.1f}[%] time={:0.2f}[s]".format((1-(edges_nt.sum()/percentage_sum))*100, 
                                                              time.time()-start_time))
        
        # select a starting node
        singlyConnected=np.nonzero(edges_nt.sum(1)==1)[0]
        doublyConnected=np.nonzero(edges_nt.sum(1)==2)[0]
        if singlyConnected.shape[0]>0:
            start=randint(0,singlyConnected.shape[0]-1)
            start=singlyConnected[start]
        elif doublyConnected.shape[0]>0:
            start=randint(0,doublyConnected.shape[0]-1)
            start=doublyConnected[start]
        else:
            raise ValueError('only only nodes with more than 2 edges left?!')
        
        # get a path through the ground-truth graph
        #t_randomPath_s=time.time()
        path,path_edges,tp=randomPathOrLoop(edges_nt,start)
        path_coords=node_coords[path,:]
        #t_randomPath_e=time.time()
        #renderNodes(_lbl,path_coords,1)
        #print(path_coords)
        
        # prepare a graph needed to refine the path
        #t_graph_s=time.time()
        gr,costs,gnode_coords,corresp,sourceind,targetind=graphForRetracing(path_coords,rad,predictions,alpha,
            dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape)
        #t_graph_e=time.time()
        #print(gr)
            
        #t_shortest_path_s=time.time()
        _a,_b=graph_tool.topology.shortest_path(gr, gr.vertex(sourceind),gr.vertex(targetind), weights=costs)
        #print("_a",_a)
        #t_shortest_path_e=time.time()
        #t_rendering_s=time.time()
        retraced_v2=[gnode_coords[v] for v in _a]
        retraced_v2.pop()
        del retraced_v2[0]
        renderNodes(lbl,np.stack(retraced_v2,1).transpose(),1)
        #t_rendering_e=time.time()

        # remove from the ground truth graph the edges belonging to the path
        # that has been re-fined already
        edges_nt=np.logical_and(edges_nt,np.logical_not(path_edges))

        #print("retracing times")
        #print("random path time",t_randomPath_e-t_randomPath_s)
        #print("graph preparation time",t_graph_e-t_graph_s)
        #print("shortest path time",t_shortest_path_e-t_shortest_path_s)
        #print("rendering itme",t_rendering_e-t_rendering_s)
    return lbl

def resamplePaths2D(lbl,predictions,edges,node_coords,rad,alpha):
    # predictions must be non-negative!
    # paint the modified ground truth into lbl
    # by re-tracing the ground truth graph
    # TODO: 
    # 1) take care of connections (ignored here) between different paths
    # 2) take care of proper loop re-tracing
    
    # pre-compute whatever can be taken outside of the loop
    dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape =\
        precomputeGraphElements2D(rad,predictions,alpha)

    edges_nt=np.copy(edges)
    print(edges_nt.sum())
    while np.any(edges_nt) :
        
        print(edges_nt.sum())
        
        # select a starting node
        singlyConnected=np.nonzero(edges_nt.sum(1)==1)[0]
        doublyConnected=np.nonzero(edges_nt.sum(1)==2)[0]
        if singlyConnected.shape[0]>0:
            start=randint(0,singlyConnected.shape[0]-1)
            start=singlyConnected[start]
        elif doublyConnected.shape[0]>0:
            start=randint(0,doublyConnected.shape[0]-1)
            start=doublyConnected[start]
        else:
            raise ValueError('only only nodes with more than 2 edges left?!')
        
        # get a path through the ground-truth graph
        #t_randomPath_s=time.time()
        path,path_edges,tp=randomPathOrLoop(edges_nt,start)
        path_coords=node_coords[path,:]
        #t_randomPath_e=time.time()
        #renderNodes(_lbl,path_coords,1)
        #print(path_coords)
        
        # prepare a graph needed to refine the path
        #t_graph_s=time.time()
        gr,costs,gnode_coords,corresp,sourceind,targetind=graphForRetracing2D(path_coords,rad,predictions,alpha,
            dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape)
        #t_graph_e=time.time()
        #print(gr)
            
        #t_shortest_path_s=time.time()
        _a,_b=graph_tool.topology.shortest_path(gr, gr.vertex(sourceind),gr.vertex(targetind), weights=costs)
        #print("_a",_a)
        #t_shortest_path_e=time.time()
        #t_rendering_s=time.time()
        retraced_v2=[gnode_coords[v] for v in _a]
        retraced_v2.pop()
        del retraced_v2[0]
        renderNodes2D(lbl,np.stack(retraced_v2,1).transpose(),1)
        #t_rendering_e=time.time()

        # remove from the ground truth graph the edges belonging to the path
        # that has been re-fined already
        edges_nt=np.logical_and(edges_nt,np.logical_not(path_edges))

        #print("retracing times")
        #print("random path time",t_randomPath_e-t_randomPath_s)
        #print("graph preparation time",t_graph_e-t_graph_s)
        #print("shortest path time",t_shortest_path_e-t_shortest_path_s)
        #print("rendering itme",t_rendering_e-t_rendering_s)
    return lbl

# this is a slower, reference implementation of graphForRetracing
# todo: connections to other traces
def graphForRetracing_reference(path_coords,rad,pred,alpha):
    # this function generates a graph used to re-trace a single path
    # path_coords nX3 table of coordinates of path points
    # rad - a scalar, how much the new path can deviate from the old one
    # pred - a prediction volume - re-tracing cost will be calculated from it
    
    # prepare coordinates of the nodes corresponding to a single node in the path
    cost_shape=np.array(pred.shape)
    diam=2*rad+1
    npath=len(path_coords)
    nlogcost=-alpha*np.log(pred)
    
    gr=graph_tool.Graph()
    edge_costs=gr.new_edge_property('double')
    node_coords=gr.new_vertex_property('vector<int>')
    node_cost=gr.new_vertex_property('double')
    path_corr=gr.new_vertex_property('int')
    s=gr.add_vertex() # source node
    node_coords[s]=np.zeros(3)
    path_corr[s]=-1
    node_cost[s]=0.0
    nodes_prev=[s]
    nodes_curr=[]
    p=0
    # nodes corresp to the first path node
    for i in range(-rad,rad+1):
        x=path_coords[p,0]+i
        if x<0 or x>=cost_shape[0]: continue
        for j in range(-rad,rad+1):
            y=path_coords[p,1]+j
            if y<0 or y>=cost_shape[1]: continue
            for k in range(-rad,rad+1):
                z=path_coords[p,2]+k
                if z<0 or z>=cost_shape[2]: continue
                v=gr.add_vertex()
                node_coords[v]=np.array([x,y,z])
                dist2path2=i*i+j*j+k*k
                node_cost[v]=nlogcost[x,y,z]+dist2path2
                path_corr[v]=p
                # connect v to other nodes corresp to the same path node
                for u in nodes_curr:
                    d2=((np.array(node_coords[v])-np.array(node_coords[u]))**2).sum()
                    if d2<=3:
                        e1=gr.add_edge(u,v)
                        e2=gr.add_edge(v,u)
                        edge_costs[e1]=node_cost[v]
                        edge_costs[e2]=node_cost[u]
                nodes_curr.append(v)
                # connect v to the source node
                e=gr.add_edge(s,v)
                edge_costs[e]=node_cost[v]
    
    # nodes corresp to the following path nodes
    p_pos=path_coords[0,:]
    for p in range(1,npath):
        nodes_prev=nodes_curr
        nodes_curr=[]
        prev_p_pos=p_pos
        p_pos=path_coords[p,:]
        pdiff=p_pos-prev_p_pos # direction of the path
        for i in range(-rad,rad+1):
            x=path_coords[p,0]+i
            if x<0 or x>=cost_shape[0]: continue
            for j in range(-rad,rad+1):
                y=path_coords[p,1]+j
                if y<0 or y>=cost_shape[1]: continue
                for k in range(-rad,rad+1):
                    z=path_coords[p,2]+k
                    if z<0 or z>=cost_shape[2]: continue
                    v=gr.add_vertex()
                    node_coords[v]=np.array([x,y,z])
                    dist2path2=i*i+j*j+k*k
                    node_cost[v]=nlogcost[x,y,z]+dist2path2
                    path_corr[v]=p
                    # connect v to other nodes corresponding to the same path node
                    for u in nodes_curr:
                        d2=((np.array(node_coords[v])-np.array(node_coords[u]))**2).sum()
                        if d2<=3:
                            e1=gr.add_edge(u,v)
                            e2=gr.add_edge(v,u)
                            edge_costs[e1]=node_cost[v]
                            edge_costs[e2]=node_cost[u]
                    nodes_curr.append(v)
                    # connect v to nodes corresponding to the previous path node
                    for u in nodes_prev:
                        ndiff=np.array(node_coords[v])-np.array(node_coords[u])
                        dotpr=(pdiff*ndiff).sum()
                        d2=(ndiff**2).sum()
                        if d2<=3 and dotpr>=0:
                            e=gr.add_edge(u,v)
                            edge_costs[e]=node_cost[v]
    # the target vertex    
    t=gr.add_vertex()
    node_coords[t]=np.zeros(3)
    path_corr[t]=-1
    node_cost[t]=0.0
    for u in nodes_curr:
        e=gr.add_edge(u,t)
        edge_costs[e]=0
  
    return gr, edge_costs,node_coords,path_corr,gr.vertex_index[s],gr.vertex_index[t]

def verifyGraphs(lbl,predictions,edges,node_coords,rad,a):
    # this function is used to compare graphs yield by the graphForRetracing
    # and the graphForRetracing_reference functions
    # it generates paths like the resampleNeurons function
    # then generates the graphs for retracing using the two functions
    # and ensures they are isomorphic
    
    # pre-compute whatever can be taken outside of the loop
    dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape =\
        precomputeGraphElements(rad,predictions,a)

    edges_nt=np.copy(edges)
    while np.any(edges_nt) :
        # select a starting node
        singlyConnected=np.nonzero(edges_nt.sum(1)==1)[0]
        doublyConnected=np.nonzero(edges_nt.sum(1)==2)[0]
        if singlyConnected.shape[0]>0:
            start=randint(0,singlyConnected.shape[0]-1)
            start=singlyConnected[start]
        elif doublyConnected.shape[0]>0:
            start=randint(0,doublyConnected.shape[0]-1)
            start=doublyConnected[start]
        else:
            raise ValueError('only triply connected nodes left?!')
        
        # get a path through the ground-truth graph
        t_randomPath_s=time.time()
        path,path_edges,tp=randomPathOrLoop(edges_nt,start)
        path_coords=node_coords[path,:]
        t_randomPath_e=time.time()
        
        # prepare a graph needed to refine the path
        t_graph_s1=time.time()
        gr1,costs1,gnode_coords1,corresp1,sourceind1,targetind1=graphForRetracing(path_coords,rad,predictions,a,
            dist2path,offset,doff,noff, neighb_intra_coords, neighb_inter_coords,cost_flat,cost_shape)
        t_graph_e1=time.time()
        t_graph_s2=time.time()
        gr2,costs2,gnode_coords2,corresp2,sourceind2,targetind2=graphForRetracing_reference(path_coords,rad,predictions,a)
        t_graph_e2=time.time()
        # define node labels to ease matching
        invariant1=gr1.new_vertex_property('long')
        for n in gr1.vertices():
            invariant1[n]=corresp1[n]+10000*gnode_coords1[n][0]+10000*gnode_coords1[n][1]+10000*gnode_coords1[n][2]
        invariant2=gr2.new_vertex_property('long')
        for n in gr2.vertices():
            invariant2[n]=corresp2[n]+10000*gnode_coords2[n][0]+10000*gnode_coords2[n][1]+10000*gnode_coords2[n][2]
            
        # check the graphs are "the same"
        isometric,isomap=graph_tool.topology.isomorphism(gr1, gr2, vertex_inv1=invariant1, vertex_inv2=invariant2, isomap=True)
        assert isometric, "the graphs are not isometric"

        # check the cost of the corresponding edges is almost equal
        for e1 in gr1.edges():
            e2=gr2.edge(isomap[e1.source()],isomap[e1.target()])
            msg=str(costs1[e1])+' != '+str(costs2[e2])
            costdiff=(costs1[e1]-costs2[e2])/(max(costs1[e1],costs2[e2])+sys.float_info.epsilon)
            assert costdiff<1e-6, msg
    
        # remove from the ground truth graph the edges belonging to the path
        # that has been re-fined already
        edges_nt=np.logical_and(edges_nt,np.logical_not(path_edges))

        print("retracing times")
        print("random path time",t_randomPath_e-t_randomPath_s)
        print("graph preparation time",t_graph_e1-t_graph_s1, t_graph_e2-t_graph_s2)
    return lbl
