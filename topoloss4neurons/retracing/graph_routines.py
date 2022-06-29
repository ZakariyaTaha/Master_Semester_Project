import numpy as np
import math
import sys
import scipy.sparse.csgraph as g
from random import randint

def graph_from_3D_mask(vol,connectivity=26):
  # noonzero value is a foreground label
  # two voxels are connected if they are neighbors
  # 26-neighborhood is used
  # pay attention that a turning structure can be rendered in two ways
  # even in 2D, resulting in different graphs:
  #  like that:
  #   OO
  #   O
  #  or like that:
  #    O
  #   O
  #
  inds=np.nonzero(vol) 
  inds=np.stack(inds,axis=-1)
  i=np.expand_dims(inds,1)
  j=np.copy(i)
  j=np.swapaxes(j,0,1)
  d2=np.power(i-j,2).sum(2)
  if connectivity==26:
    t=3
  elif connectivity==18:
    t=2
  elif connectivity==6:
    t=1
  edges=np.logical_and(d2<=t+sys.float_info.epsilon,d2>0+sys.float_info.epsilon)
  return inds,edges,d2

def graph_from_2D_mask(img,connectivity=8):
  # noonzero value is a foreground label
  # two voxels are connected if they are neighbors
  # pay attention that a turning structure can be rendered in two ways
  # even in 2D, resulting in different graphs:
  #  like that:
  #   OO
  #   O
  #  or like that:
  #    O
  #   O
  #
  inds=np.nonzero(img) 
  inds=np.stack(inds,axis=-1)
  i=np.expand_dims(inds,1)
  j=np.copy(i)
  j=np.swapaxes(j,0,1)
  d2=np.power(i-j,2).sum(2)
  if connectivity==8:
    t=2
  elif connectivity==4:
    t=1
  edges=np.logical_and(d2<=t+sys.float_info.epsilon,d2>0+sys.float_info.epsilon)
  return inds,edges,d2

def cycleBasis(edges):
  # get a basis of cycles for a graph
  # not the fastest, because some operations are repeated in g.depth_first_order
  visited=np.zeros(edges.shape[0])
  cycles=[]
  nodes, parent = g.depth_first_order(edges,0)
  for n in nodes:
    children=np.nonzero(edges[n,:])
    children=children[0]
    for c in children:
      if c==parent[n]:
        continue
      if visited[c]==1:
        new_cycle=[]
        #backtrack
        m=n
        while m!=c:
          new_cycle.append(m)
          m=parent[m]
        new_cycle.append(m)
        cycles.append(new_cycle)
    visited[n]=1
  return cycles
        
def reduceCycleBasis(cycles,n):
  # n is the node number
  # reduces the cycle basis to minimal cycles
  # (or attempts to :D)
  # NOTE: it contains a "poor man's" version of minimal cycle basis extraction
  # it does work for the cycle bases returned from the DFS traversal
  # but does not work on general graphs
  cnum=len(cycles)
  basis=[]
  for i in range(cnum):
    basis.append(np.zeros((n,n)).astype(np.bool))
    prevn=cycles[i][-1]
    for node in cycles[i]:
      basis[i][prevn,node]=True
      prevn=node
    basis[i]=basis[i].reshape(n**2)

  # reduce basis - be careful, not a general algorithm
  changed=True
  while changed:
    changed=False
    # try to reduce all pairs of basis cycles
    for i in range(cnum-1,-1,-1):
      for j in range(i):
        xor=np.logical_xor(basis[i],basis[j])
        if xor.sum()<basis[i].sum():
          changed=True
          basis[i]=xor
          xor=np.logical_xor(basis[i],basis[j])
        if xor.sum()<basis[j].sum():
          changed=True
          basis[j]=xor
  return basis

def minCycleBasis(edges):
  cycles=cycleBasis(edges)
  minbasis=reduceCycleBasis(cycles,edges.shape[0])
  return minbasis

def detectClusters(edges):
  clusters=[]
  for k in range(edges.shape[0]):
    cluster=[k]
    last_added=[k]
    while len(last_added)>0:
      candidates=list(np.nonzero(edges[np.array(last_added)])[1])
      last_added=[]
      while len(candidates)>0:
        c=candidates.pop()
        if c in cluster:
          continue
        if np.all(edges[c][np.array(cluster)]):
          cluster.append(c)
          last_added.append(c)
    if len(cluster)>2:
      clusters.append(cluster)
  return clusters

def removeACluster(edges,nodes,cluster):
  # check if we can find a node that can be removed without altering the connectivity
  border=list(np.nonzero(edges[np.array(cluster)])[1])
  for n in cluster:
    cluster_n=cluster.copy()
    cluster_n.remove(n)
    border_n=list(np.nonzero(edges[np.array(cluster_n)])[1])
    if set(border_n)==set(border): # remove the node
      edges=np.delete(edges,(n),axis=0)
      edges=np.delete(edges,(n),axis=1)
      nodes=np.delete(nodes,(n),axis=0)
      cluster.remove(n)
      return edges,nodes
  # if no node can be removed, try removing edges
  # make sure you do not diisconnect the border - true as long as in a cluster
  # make sure you do not leave a single dangling voxel - true because remove node would have removed a cluster node that would be dangling after edge removal
  # make sure you are not creating a cycle - ok, because the cycles will be clusters too
  # so you can remove any edge
  edges[cluster[0],cluster[1]]=0
  edges[cluster[1],cluster[0]]=0
  
  return edges,nodes

def detectAndRemoveACluster(edges,nodes):
  clusters=[]
  for k in range(edges.shape[0]):
    cluster=[k]
    last_added=[k]
    while len(last_added)>0:
      candidates=list(np.nonzero(edges[np.array(last_added)])[1])
      last_added=[]
      while len(candidates)>0:
        c=candidates.pop()
        if c in cluster:
          continue
        if np.all(edges[c][np.array(cluster)]):
          cluster.append(c)
          last_added.append(c)
    if len(cluster)>2: #found a cluster!
      edges,nodes=removeACluster(edges,nodes,cluster)
      return edges,nodes,True
  return edges,nodes,False # no cluster found

def detectAndRemoveClusters(edges,nodes=None):
  nodes=nodes if nodes else np.array(range(edges.shape[0]))
  res=True
  ncluster=-1
  while res:
    ncluster+=1
    edges,nodes,res=detectAndRemoveACluster(edges,nodes)
  print("there were {} clusters".format(ncluster))
  return edges,nodes  

def verifyEdges(edges,inds,dist=3):
  # check that all connected nodes are within a distance of sqrt(3)
  i=np.expand_dims(inds,1)
  j=np.copy(i)
  j=np.swapaxes(j,0,1)
  d2=np.power(i-j,2).sum(2)
  return np.all(edges*d2<=dist+sys.float_info.epsilon)

