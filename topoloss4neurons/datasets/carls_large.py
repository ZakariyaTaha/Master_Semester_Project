from collections import namedtuple
import os
import numpy as np
#from skimage.external import tifffile
from scipy.ndimage.morphology import distance_transform_edt as dist
import time
import tifffile
import torch

def findCubes(brain_i):
    xs, ys, zs = getCubeCoords(brain_i)
    coords = getCoords(brain_i)
    cubes = []
    for c in coords:
        xi = np.where(c[0] > np.array(xs))[0][-1]
        yi = np.where(c[1] > np.array(ys))[0][-1]
        zi = np.where(c[2] > np.array(zs))[0][-1]
        if [xi,yi,zi] not in cubes:
            cubes.append([xi,yi,zi])
    return cubes

def getCubeCoords(brain_i):
    """
    Returns xyz borders of the cubes according to terafiles
    """
    if brain_i == 6:
        direc = "/datasets/6res11711x16382x2000/"
    elif brain_i == 8:
        direc = "/datasets/8res11692x19566x1600/"
    elif brain_i == 9:
        direc = "/datasets/9res11692x16123x1700/"
    else:
        return
    yind = sorted([int(x)//10 for x in os.listdir(direc) if x.startswith("0") or x.startswith("1")] )
    xind = sorted([int(x.split("_")[-1])//10 for x in os.listdir(os.path.join(direc,"000000")) if x.startswith("0") or x.startswith("1")] )
    zind = sorted([int(x.split("_")[-1][:-4])//10 for x in os.listdir(os.path.join(direc,"000000", "000000_000000")) if x.startswith("0") or x.startswith("1")] )
    
    return xind, yind, zind

def getCube(brain_i, cube):
    """
    Returns the image of the desired cub in the desired brain
    """
    if brain_i == 6:
        direc = "/datasets/6res11711x16382x2000/"
    elif brain_i == 8:
        direc = "/datasets/8res11692x19566x1600/"
    elif brain_i == 9:
        direc = "/datasets/9res11692x16123x1700/"
    else:
        return
    xs,ys,zs = getCubeCoords(brain_i)
    x,y,z = cube
    l1 = "{:06d}".format(ys[y]*10)
    l2 = "{:06d}_{:06d}".format(ys[y]*10,xs[x]*10)
    l3 = "{:06d}_{:06d}_{:06d}.tif".format(ys[y]*10,xs[x]*10,zs[z]*10)
    volume = tifffile.imread(os.path.join(direc,l1,l2,l3))
    return volume.transpose((1,2,0))

def getLabel(brain_i,cube):
    """
    Returns the label of the desired cube in the desired brain
    """
    xs, ys, zs = getCubeCoords(brain_i)
    xi, xf = xs[cube[0]:cube[0]+2]
    yi, yf = ys[cube[1]:cube[1]+2]
    zi, zf = zs[cube[2]:cube[2]+2]
    m=[1,1,1]
    o=[-int(yi),-int(xi),-int(zi)]
    scale_factor=[1,1,1]

    downsampling=torch.tensor(scale_factor,dtype=torch.double)
    offset=torch.tensor(o,dtype=torch.double)
    scale=torch.tensor(m,dtype=torch.double)
    volDims=torch.tensor([3,2,4],dtype=torch.long)
    vols = np.zeros((yf-yi,xf-xi,zf-zi), dtype=np.uint8)
    if brain_i == 6:
        renderSWC2volume(swcname1, volDims, vols, scale, offset, downsampling)
    elif brain_i == 8:
        renderSWC2volume(swcname2, volDims, vols, scale, offset, downsampling)
    elif brain_i == 9:
        renderSWC2volume(swcname3, volDims, vols, scale, offset, downsampling)
    else:
        return
    return vols

def readSWC(brain_i):
    """
    Returns the nodes in the swc file
    """
    if brain_i == 6:
        swcfname = "/datasets/expcodes/AL066-AL_stamp_2019_07_23_10_34.ano.eswc"
    elif brain_i == 8:
        swcfname = "/datasets/expcodes/AL080_stamp_2020_01_22_13_26.ano.eswc"
    elif brain_i == 9:
        swcfname = "/datasets/expcodes/AL092_stamp_2020_01_09_11_10.ano.eswc_sorted.eswc"
    else:
        return
    nodes=dict()
    for a in open(swcfname):
        if (re.match('\s*\#',a)!=None):
#             print("commment line", a)
            continue
        b=a.split()
        c=map(lambda x: float(x), b)
        d=list(c)
        nodes[int(d[0])]=d
    return nodes

def getCoords(brain_i):
    """
    Returns location of each positive pixels
    """
    nodes = readSWC(brain_i)
    coords = []
    for k in nodes :
        n=nodes[k]
        if brain_i == 6:
            x,y,z= volInds(n,[1,1,1],[2,3,4], [0,0,0], [1,1,1])
        if brain_i == 8:
            x,y,z= volInds(n,[1,1,1],[2,3,4], [0,0,0], [1,1,1])
        if brain_i == 9:
            x,y,z= volInds(n,[1,1,1],[2,3,4], [0,0,0], [1,1,1])
        coords.append([x,y,z])
    return coords

def renderSWC2volume(swcfname, volumeDims, volCL, scale, offset, downsampling):
    '''
      swcfname      name of the swc file
      volumeDims    one-dimensional array;
                    volumeDims[1] is index of volCL dimension corresponding to X
                    volumeDims[2] is index of volCL dimension corresp to Y
                    volumeDims[3] is index of volCL dimension corresp to Z
                    X,Y,Z are as interpreted in the CWS format
      volCL         np array into which we will render ground truth centerlines
    '''
    distthresh=40
    nodes=dict()
    for a in open(swcfname):
        if (re.match('\s*\#',a)!=None):
#             print("commment line", a)
            continue
        b=a.split()
        c=map(lambda x: float(x), b)
        d=list(c)
        nodes[int(d[0])]=d
        #print(d)
    # start here
    for k in nodes :
        n=nodes[k]
        x,y,z= volInds(n,scale,volumeDims, offset, downsampling)
        parent=nodes.get(int(n[6]),None)
        #print(n)
        if parent!=None :
            #print(n,parent)
            xp,yp,zp=volInds(parent,scale,volumeDims, offset, downsampling)
            #print(x,y,z,volCL.shape)
            #print(xp,yp,zp,volCL.shape)
            if (x!=xp or y!=yp or z!=zp) and ((abs(x-xp)+abs(y-yp)+abs(z-zp))<distthresh):
                #print("line: ({},{},{})-({},{},{})".format(xp,yp,zp,x,y,z))
                traceLine(volCL,np.array([xp,yp,zp]),np.array([x,y,z]))

def volInds(swcCoords,scale,volDims,offset,downsampling):
    x=int((swcCoords[volDims[0]]*scale[0]+offset[0])*downsampling[0])
    y=int((swcCoords[volDims[1]]*scale[1]+offset[1])*downsampling[1])
    z=int((swcCoords[volDims[2]]*scale[2]+offset[2])*downsampling[2])
    return x,y,z

def traceLine(lbl,begPoint,endPoint):
    # endPoint and begPoint should be np.arrays
    # lbl is an np.array to which the line is rendered
    d=endPoint-begPoint
    s=begPoint
    mi=np.argmax(np.fabs(d))
    coef=d/d[mi]
    sz=np.array(lbl.shape)
    numsteps=int(abs(d[mi]))+1
    step=int(d[mi]/abs(d[mi]))
    for t in range(0,numsteps):
        pos=np.array(s+coef*t*step)
        if np.all(pos<sz) and np.all(pos>=0):
            #print(pos)
            lbl[tuple(pos.astype(np.int))]=1
#         else:
#             print("reqested point",pos,"but the volume size is",sz)
    return lbl

# ===================================================================================
base = ""
# base_graphs = "/cvlabdata2/cvlab/datasets_leo/isbi12_em/graphs_isbi12/lbl_graph/train"
# path_images={"train":os.path.join(base, 'full_data'),
#              "test" :os.path.join(base, 'full_data'),
#              "full" :os.path.join(base, 'full_data')}
# path_test_images={"orig":os.path.join(base, 'imagery_test'),
#              "half" :os.path.join(base, 'imagery_test')}
# path_labels={"orig":os.path.join(base, 'masks_thick'),
#              "half" :os.path.join(base, 'masks_thick')}
# path_labels_thin={"orig":os.path.join(base, 'masks'),
#                   "half" :os.path.join(base, 'masks')}
# path_labels_dist={"train":os.path.join(base, 'dist_lbl'),
#                   "test" :os.path.join(base, 'dist_lbl')}
train_names = list(np.load("/datasets/expcodes/trainCubes.npy"))
test_names = list(np.load("/datasets/expcodes/testCubes.npy"))

sequences = {"training": train_names,
             "testing":  test_names,
             "trial":    train_names[:20],
             "all":      train_names+test_names}
s = time.time()
# ===================================================================================
DataPoint = namedtuple("DataPoint", ["image", "dist_labels"])

cubes1 = findCubes(6)
cubes2 = findCubes(8)
cubes3 = findCubes(9)
cubes = [cubes1, cubes2, cubes3]
def _data_point(fid, size="orig", labels="all", graph=False, threshold=20):
    print(fid)
    bi = int(fid.split("_")[0])
    ci = int(fid.split("_")[-1][:-4])
    
    image = getCube(bi, cubes[bi][ci]) / 32767
    label = getLabel(bi, cubes[bi][ci])
    print(np.sum(label))

    d_label = dist(1-label)
    d_label[d_label>threshold] = threshold

    return DataPoint(image, d_label)

def load_dataset(sequence='training', size="orig", labels="all", each=1, graph=False, threshold=20):

    data_points = tuple(_data_point(fid, size, labels, graph, threshold) for fid in sequences[sequence][::each])

    return data_points
