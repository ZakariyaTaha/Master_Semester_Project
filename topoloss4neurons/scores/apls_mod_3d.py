#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017

@author: avanetten

"""

# from __future__ import print_function
# from . import apls_utils
# from . import apls_plots
# from . import osmnx_funcs
# import sp_metric
# import topo_metric
# import apls_tools
# import graphTools
import networkx as nx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import utm           # pip install utm
import copy
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import time
import os
import sys
import scipy
import argparse
# import pandas as pd
import shapely.wkt
# import osmnx as ox   # https://github.com/gboeing/osmnx
# import pickle
# import shutil

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
print("path_apls:", path_apls)
# add path and import graphTools
sys.path.append(path_apls_src)
import apls_utils_3d
# import apls_plots
#import osmnx_funcs
#import graphTools
#import wkt_to_G
#import topo_metric
#import sp_metric

# if in docker, the line below may be necessary
matplotlib.use('agg')

def get_length(line):
    return scipy.spatial.distance.euclidean(line.coords[0], line.coords[1])

def get_bounds(line):
    locs = list(zip(line.coords[0], line.coords[1]))
    return np.concatenate((np.min(locs, 1),np.max(locs, 1)))

def project(line, point):
    x = np.array(line.coords[1])-np.array(line.coords[0])
    y = np.array(point.coords[0]) - np.array(line.coords[0])
    if np.all(np.array(point.coords[0]) == np.array(line.coords[0])):
        return 0
    if t(line,point) >= 0 and t(line,point) <= (np.sqrt(np.dot(x,x)) / np.sqrt(np.dot(y,y))):
        cp = line.coords[0] + x*t(line,point)/np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y))
    elif t(line,point) < 0:
        cp = line.coords[0]
    else:
        cp = line.coords[1]
    proj = scipy.spatial.distance.euclidean(line.coords[0], cp)
    return proj

def t(line, point):
    x = np.array(line.coords[1])-np.array(line.coords[0])
    y = np.array(point.coords[0]) - np.array(line.coords[0])
    cosxy = np.dot(y,x)/(np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))
    return cosxy

def get_distance(line, point):
    x = np.array(line.coords[1])-np.array(line.coords[0])
    y = np.array(point.coords[0]) - np.array(line.coords[0])
    if t(line,point) >= 0 and t(line,point) <= (np.sqrt(np.dot(x,x)) / np.sqrt(np.dot(y,y))):
        cp = line.coords[0] + x*t(line,point)/np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y))
        dist = scipy.spatial.distance.euclidean(cp, point)
    elif t(line,point) < 0:
        dist = scipy.spatial.distance.euclidean(line.coords[0], point)
    else:
        dist = scipy.spatial.distance.euclidean(line.coords[1], point)
    return dist

def get_line_xyz(line):
    point0 = line.coords[0]
    point1 = line.coords[1]
    return (np.array('d', [point0[0], point1[0]]), np.array('d', [point0[1], point1[1]]), np.array('d', [point0[2], point1[2]]))
    
def get_point_xyz(point):
    return (np.array('d', [point.x]), np.array('d', [point.y]), np.array('d', [point.z]))
    
    
def check_add_geometry(G_):

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if 'geometry' not in data:

            sourcex, sourcey, sourcez = G_.nodes[u]['x'],  G_.nodes[u]['y'],  G_.nodes[u]['z']
            targetx, targety, targetz = G_.nodes[v]['x'],  G_.nodes[v]['y'],  G_.nodes[v]['z']
            line_geom = LineString([Point(sourcex, sourcey, sourcez),
                                    Point(targetx, targety, targetz)])
            data['geometry'] = line_geom
            data['length'] = get_length(line_geom)

    return G_

###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.

    Notes
    -----
    If identical edges exist, remove extras.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    # G_ = G_.copy()
    # for i,(u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # create linestring if no geometry reported

        if 'geometry' not in data:
            sourcex, sourcey, sourcez = G_.nodes[u]['x'],  G_.nodes[u]['y'],  G_.nodes[u]['z']
            targetx, targety, targetz = G_.nodes[v]['x'],  G_.nodes[v]['y'],  G_.nodes[v]['z']
            line_geom = LineString([Point(sourcex, sourcey, sourcez),
                                    Point(targetx, targety, targetz)])
            data['geometry'] = line_geom
            data['length'] = get_length(line_geom)

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            # G_.edges[u][v]['geometry'] = lstring
        else:
            # check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            # print (u,v,key,"create_edge_linestrings() line_geom:", line_geom)
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y'], G_.nodes[u]['z']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y'], G_.nodes[v]['z']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                # data['geometry'].coords = list(line_geom.coords)[::-1]
                data['geometry'] = line_geom_rev
            # else:
            #    continue

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))  # , key))
                            if verbose:
                                print("\nRedundant edge:", u, v)  # , key)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)  # , key)
            # # for (u,v,key) in bad_edges:
            # try:
            #     G_.remove_edge(u, v)  # , key)
            # except:
            #     if verbose:
            #         print("Edge DNE:", u, v)  # ,key)
            #     pass

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    '''
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= get_length(line):
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = i*get_length(line)#project(line, Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y, cp.z)]),
                LineString([(cp.x, cp.y, cp.z)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y, cp.z)]),
        LineString([(cp.x, cp.y, cp.z)] + coords[i:])]


###############################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):
    """
    Return closest edge to point, and distance to said edge.

    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # print((" in get_closest_edge(): u,v,key,data:", u,v,key,data))
        # print ("  in get_closest_edge(): data:", data)

        # skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        if verbose:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(get_distance(line, p))
        edge_list.append([u, v, key])
    # get closest edge
    min_idx = np.argmin(np.array(dist_list))
    min_dist = dist_list[min_idx]

    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


###############################################################################
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5, dist_close_node=10,
                        nearby_nodes_set=set([]), allow_renaming=False,
                        verbose=False, super_verbose=False):
    """
    Insert a new node in the graph closest to the given point.

    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    # check if node_id already exists in G
    # if node_id in set(G_.nodes()):
    #    print ("node_id:", node_id, "already in G, cannot insert node!")
    #    return

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set,
            verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if verbose:
        print("Inserting point:", node_id)
        print("best edge:", best_edge)
        print("  best edge dist:", min_dist)
        u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y'], G_.nodes[u]['z']]
        v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y'], G_.nodes[v]['z']]
        print("ploc:", (point.x, point.y, point.z))
        print("uloc:", u_loc)
        print("vloc:", v_loc)

    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            if verbose:
                print("Node ID:", node_id, "already exists, skipping...")
            return G_, {}, -1, -1, -1

        # G_.edges[best_edge[0]][best_edge[1]][0]['geometry']
        line_geom = best_geom

        # Length along line that is closest to the point
        line_proj = project(line_geom, point)
        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(project(line_geom, point))
        x, y, z = new_point.x, new_point.y, new_point.z

        #################
        # create new node

        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        # set properties
        # props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                      'lat':     lat,
                      'lon':     lon,
                      'osmid':   node_id,
                      'x':       x,
                      'y':       y,
                      'z':       z}
        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])
        # remove extraneous 0 key

        # print ("edge_props_new.keys():", edge_props_new)
        # if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]

        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        # line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line is None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", get_length(line_geom))
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, 0, 0, 0

        if verbose:
            print("split_line:", split_line)

        # if cp.is_empty:
        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y, outnode_z = -1, -1, -1
            x_p, y_p, z_p = new_point.x, new_point.y, new_point.z
            x_u, y_u, z_u = G_.nodes[u]['x'], G_.nodes[u]['y'], G_.nodes[u]['z']
            x_v, y_v, z_v = G_.nodes[v]['x'], G_.nodes[v]['y'], G_.nodes[v]['z']
            # if verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            #dist_close_node = 0.05  # meters
            if (abs(x_p - x_u) <= dist_close_node) and (abs(y_p - y_u) <= dist_close_node) and (abs(z_p - z_u) <= dist_close_node):
                outnode = u
                outnode_x, outnode_y, outnode_z = x_u, y_u, z_u
            elif (abs(x_p - x_v) <= dist_close_node) and (abs(y_p - y_v) <= dist_close_node) and (abs(z_p - z_v) <= dist_close_node):
                outnode = v
                outnode_x, outnode_y, outnode_z = x_v, y_v, z_v
            # original method with exact matching
            # if (x_p == x_u) and (y_p == y_u):
            #    outnode = u
            #    outnode_x, outnode_y = x_u, y_u
            # elif (x_p == x_v) and (y_p == y_v):
            #    outnode = v
            #    outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p, z_p:", x_p, y_p, z_p)
                print("x_u, y_u, z_u:", x_u, y_u, z_u)
                print("x_v, y_v, z_v:", x_v, y_v, z_v)
                # return
                return G_, {}, 0, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
                return Gout, node_props, x_p, y_p, z_p

            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y, outnode_z)])
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = get_length(line1)
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print("  line1.length:", get_length(line1))
                    print("  x_u, y_u, z_u :", x_u, y_u, z_u)
                    print("  x_v, y_v, z_v :", x_v, y_v, z_v)
                    print("  x_p, y_p, z_p :", x_p, y_p, z_p)
                    print("  new_point:", new_point)
                    print("  Point(outnode_x, outnode_y, outnode_z):",
                          Point(outnode_x, outnode_y, outnode_z))
                    return

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y, z

                # originally, if not renaming nodes,
                # just ignore this complication and return the orignal
                # return G_, node_props, 0, 0

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            # print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y'], G_.nodes[u]['z']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y'], G_.nodes[v]['z']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            # geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", get_length(line_geom))
                print("   line1_length:", get_length(line1))
                print("   line2_length:", get_length(line2))
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = get_length(line1)
            edge_props_line1['geometry'] = line1
            # remove geometry?
            # edge_props_line1.pop('geometry', None)
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = get_length(line2)
            edge_props_line2['geometry'] = line2
            # remove geometry?
            # edge_props_line1.pop('geometry', None)

            # insert edge regardless of direction
            # G_.add_edge(u, node_id, **edge_props_line1)
            # G_.add_edge(node_id, v, **edge_props_line2)

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            if verbose:
                print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

            # remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y, z


###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          dist_close_node=10,
                          allow_renaming=False,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y', z_coord='z',
                          verbose=True, super_verbose=False):
    """
    Wrapper around insert_point_into_G() for all control_points.

    Notes
    -----
    control_points are assumed to be of the format:
        [[node_id, x, y], ... ]

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    control_points : array
        Points to insert in the graph, assumed to the of the format:
            [[node_id, x, y], ... ]
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    n_nodes_for_kd : int
        Minumu size of graph to render to kdtree to speed node placement.
        Defaults to ``1000``.
    n_neighbors : int
        Number of neigbors to return if building a kdtree. Defaults to ``20``.
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, new_xs, new_ys : tuple
        Gout is the updated graph
        new_xs, new_ys are coordinates of the inserted points
    """

    t0 = time.time()

    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > n_nodes_for_kd:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys, new_zs = [], [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys, new_zs

    for i, [node_id, x, y, z] in enumerate(control_points):
        if verbose:
            if (i % 20) == 0:
                print(i, "/", len(control_points),
                      "Insert control point:", node_id, "x =", x, "y =", y, "z =", z)
        point = Point(x, y, z)

        # if large graph, determine nearby nodes
        if len(G_.nodes()) > n_nodes_for_kd:
            # get closest nodes
            node_names, dists_m_refine = apls_utils.nodes_near_point(
                    x, y, z, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord, 
                    z_coord=z_coord,
                    # radius_m=radius_m,
                    n_neighbors=n_neighbors,
                    verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
#         print(len(insert_point_into_G(
#             Gout, point, node_id=node_id,
#             max_distance_meters=max_distance_meters,
#             dist_close_node=dist_close_node,
#             nearby_nodes_set=nearby_nodes_set,
#             allow_renaming=allow_renaming,
#             verbose=super_verbose)))
        Gout, node_props, xnew, ynew, znew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            dist_close_node=dist_close_node,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=super_verbose)
        # xnew = node_props['x']
        # ynew = node_props['y']
        if (x != 0) and (y != 0) and (z != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)
            new_zs.append(znew)

    t1 = time.time()
    if verbose:
        print("Time to run insert_control_points():", t1-t0, "seconds")
    return Gout, new_xs, new_ys, new_zs


###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, dist_close_node=10, allow_renaming=False,
                           figsize=(0, 0),
                           verbose=False, super_verbose=False):
    """
    Insert midpoint nodes into long edges on the graph.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    figsize : tuple
        Figure size for optional plot. Defaults to ``(0,0)`` (no plot).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    # midpoint_loc = 0.5        # take the central midpoint for straight lines
    if len(G_.nodes()) == 0:
        return G_, [], [], []

    # midpoints
    xms, yms, zms = [], [], []
    Gout = G_.copy()
    # midpoint_name_val, midpoint_name_inc = 0.01, 0.01
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    # for u, v, key, data in G_.edges(keys=True, data=True):
    for u, v, data in G_.edges(data=True):

        # curved line
        if 'geometry' in data:

            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])
            # _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
            #                                              G_.nodes[u]['lon'])

            linelen = data['length']
            line = data['geometry']

            xs, ys, zs = get_xyz(line)  # for plotting

            #################
            # check if curved or not
            minx, miny, minz, maxx, maxy, maxz = get_bounds(line)
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny, minz], [maxx, maxy, maxz])
            # ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                # print "Line straight, skipping..."
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                # print "Line too short, skipping..."
                continue
            #################

            if verbose:
                print("create_graph_midpoints()...")
                print("  u,v:", u, v)
                print("  data:", data)
                print("  edge_props_init:", edge_props_init)

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * get_length(line)]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print("  interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp, zms_tmp = [], [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("    ", j, "interp_dist:", d)

                midPoint = line.interpolate(d)
                xm0, ym0, zm0 = get_point_xyz(midPoint)
                xm = xm0[-1]
                ym = ym0[-1]
                zm = zm0[-1]
                point = Point(xm, ym, zm)
                xms.append(xm)
                yms.append(ym)
                zms.append(zm)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                zms_tmp.append(zm)
                if verbose:
                    print("    midpoint:", xm, ym, zm)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("    node_id:", node_id)

                # if j > 3:
                #    continue

                # add to graph
                Gout, node_props, _, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    dist_close_node=dist_close_node,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

        # plot, if desired
        if figsize != (0, 0):
            fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
            ax.plot(xs, ys, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            ax.scatter(xm, ym, color='red')
            ax.set_title('Line Midpoint')
            plt.axis('equal')

    return Gout, xms, yms, zms

###############################################################################
def make_graphs(G_gt, G_p,
                weight='length',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                dist_close_node=10,
                allow_renaming=False,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground truth and propsal graphs, and get paths.

    Notes
    -----
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time'``.
    max_nodes_for_midpoints : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    linestring_delta : float
        Distance in meters between linestring midpoints.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``0.012``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime
    """

    t0 = time.time()
    print("Executing make_graphs()...")
    '''
    print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)
    '''
    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt.to_undirected())
    G_gt = check_add_geometry(G_gt)
    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    # G_gt_cp = create_edge_linestrings(G_gt.to_undirected())

    if verbose:
        print("len G_gt.nodes():", len(list(G_gt0.nodes())))
        print("len G_gt.edges():", len(list(G_gt0.edges())))

    if verbose:
        print("Creating gt midpoints")
    G_gt_cp, xms, yms, zms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        dist_close_node=dist_close_node,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y, u_z = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y'], G_gt_cp.nodes[n]['z']
        control_points_gt.append([n, u_x, u_y, u_z])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))

    # get ground truth paths
    if verbose:
        print("Get ground truth paths...")
    all_pairs_lengths_gt_native = dict(
        nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight))
    ###############

    ###############
    # get proposal graph with native midpoints
    '''
    print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)
    '''
    G_p = create_edge_linestrings(G_p.to_undirected())

    if verbose:
        print("len G_p.nodes():", len(G_p.nodes()))
        print("len G_p.edges():", len(G_p.edges()))

    if verbose:
        print("Creating proposal midpoints")
    G_p_cp, xms_p, yms_p, zms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        dist_close_node=dist_close_node,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)

    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("len G_p_cp.edges():", len(G_p_cp.edges()))

    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y, u_z = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y'], G_p_cp.nodes[n]['z']
        control_points_prop.append([n, u_x, u_y, u_z])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.all_pairs_dijkstra_path_length(G_p_cp, weight=weight))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("G_p.nodes():", G_p.nodes())
    G_p_cp_prime, xn_p, yn_p, zn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist,
        dist_close_node=dist_close_node,
        allow_renaming=allow_renaming,
        verbose=super_verbose, super_verbose=super_verbose)

#    G_p_cp, xn_p, yn_p = insert_control_points(G_p_cp, control_points_gt,
#                                        max_distance_meters=max_snap_dist,
#                                        allow_renaming=allow_renaming,
#                                        verbose=verbose)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt, zn_gt = insert_control_points(
        G_gt,
        control_points_prop,
        max_distance_meters=max_snap_dist,
        dist_close_node=dist_close_node,
        allow_renaming=allow_renaming,
        verbose=super_verbose, super_verbose=super_verbose)

    ###############
    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight))

    tf = time.time()
    print("Time to run make_graphs in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def is_intersection(G,n):
    if len(G.edges(n))>=3:
        return True
    return False

def make_graphs_yuge(G_gt, G_p,
                     weight='length',
                     max_nodes=500,
                     max_snap_dist=4,
                     dist_close_node=10,
                     allow_renaming=False,
                     select_intersections=False,
                     verbose=True, super_verbose=False):
    """
    Match nodes in large ground truth and propsal graphs, and get paths.

    Notes
    -----
    Skip midpoint injection and only select a subset of routes to compare.
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time'``.
    max_nodess : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime
    """

    t0 = time.time()
    if verbose:
        print("Executing make_graphs_yuge()...")
    '''
    print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)
    '''
    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    # G_gt_cp = create_edge_linestrings(G_gt.to_undirected())
    if verbose:
        print("len(G_gt.nodes()):", len(G_gt_cp.nodes()))
        print("len(G_gt.edges()):", len(G_gt_cp.edges()))
        # print("G_gt.nodes():", G_gt_cp.nodes())
        # print("G_gt.edges()):", G_gt_cp.edges())
        # gt node and edge props
        node = random.choice(list(G_gt.nodes()))
        print("node:", node, "G_gt random node props:", G_gt.nodes[node])
        edge_tmp = random.choice(list(G_gt.edges()))
        print("G_gt edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0], edge_tmp[1], 0])
        # prop node and edge props
        node = random.choice(list(G_p.nodes()))
        print("node:", node, "G_p random node props:", G_p.nodes[node])
        edge_tmp = random.choice(list(G_p.edges()))
        print("G_p edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0], edge_tmp[1], 0])

    if select_intersections:
        sel_nodes_gt = []
        for n in G_gt_cp.nodes():
            if is_intersection(G_gt_cp, n):
                sel_nodes_gt.append(n)
    else:
        sel_nodes_gt = list(G_gt_cp.nodes())

    # get ground truth control points, which will be a subset of nodes
    sample_size = min(max_nodes, len(sel_nodes_gt))
    rand_nodes_gt = random.sample(sel_nodes_gt, sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)

    control_points_gt = []
    for n in rand_nodes_gt:
        u_x, u_y, u_z = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y'], G_gt_cp.nodes[n]['z']
        control_points_gt.append([n, u_x, u_y, u_z])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))

    # get route lengths between all control points
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_gt_native...")
    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               sample_size, "nodes:", time.time() - tt, "seconds"))

    # get individual routes (super slow!)
    #t0 = time.time()
    #all_pairs_lengths_gt_native = {}
    # for source in rand_nodes_gt:
    #    print ("source:", source)
    #    source_dic = {}
    #    for target in rand_nodes_gt:
    #        print ("target:", target)
    #        p = nx.dijkstra_path_length(G_gt_init, source, target, weight=weight)
    #        source_dic[target] = p
    #    all_pairs_lengths_gt_native[source] = source_dic
    #print ("Time to compute all source routes:", time.time() - t0, "seconds")
    ## ('Time to compute all source routes:', 9.418055057525635, 'seconds')

    #all_pairs_lengths_gt_native = nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight)
    ###############

    ###############
    # get proposal graph with native midpoints
    G_p_cp = G_p.to_undirected()
    #G_p_cp = create_edge_linestrings(G_p.to_undirected())
    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("G_p_cp.edges():", len(G_p_cp.edges()))

    if select_intersections:
        sel_nodes_p = []
        for n in G_p_cp.nodes():
            if is_intersection(G_p_cp, n):
                sel_nodes_p.append(n)
    else:
        sel_nodes_p = list(G_p_cp.nodes())

    # get control points, which will be a subset of nodes
    # (original method sets proposal control points as all nodes in G_p_cp)
    sample_size = min(max_nodes, len(sel_nodes_p))
    rand_nodes_p = random.sample(sel_nodes_p, sample_size)
    rand_nodes_p_set = set(rand_nodes_p)

    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y, u_z = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y'], G_p_cp.nodes[n]['z']
        control_points_prop.append([n, u_x, u_y, u_z])

    # get paths
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_prop_native...")
    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("len G_p.nodes():", len(G_p.nodes()))
    G_p_cp_prime, xn_p, yn_p, zn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        dist_close_node=dist_close_node,
        allow_renaming=allow_renaming, verbose=super_verbose, super_verbose=super_verbose)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt, zn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        dist_close_node=dist_close_node,
        allow_renaming=allow_renaming, verbose=super_verbose, super_verbose=super_verbose)

    ###############
    # get paths for graphs_prime
    # gather all paths from nodes of interest, keep only routes to control nodes
    # gt_prime
    tt = time.time()
    all_pairs_lengths_gt_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_gt_prime...")
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
#    for source in G_gt_cp_prime_nodes_set:
#        if source in G_gt_cp_prime_nodes_set:
#            paths_tmp = nx.single_source_dijkstra_path_length(G_gt_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_p_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    # prop_prime
    tt = time.time()
    all_pairs_lengths_prop_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_prop_prime...")
    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
#    for source in G_p_cp_prime_nodes_set:
#        if source in G_p_cp_prime_nodes_set:
#            paths_tmp = nx.single_source_dijkstra_path_length(G_p_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_gt_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    #all_pairs_lengths_gt_prime = nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight)
    #all_pairs_lengths_prop_prime = nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight)

    ###############
    tf = time.time()
    if verbose:
        print("Time to run make_graphs_yuge in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    """
    Compute APLS metric for single path.

    Notes
    -----
    Compute normalize path difference metric, if len_prop < 0, return diff_max

    Arguments
    ---------
    len_gt : float
        Length of ground truth edge.
    len_prop : float
        Length of proposal edge.
    diff_max : float
        Maximum value to return. Defaults to ``1``.

    Returns
    -------
    metric : float
        Normalized path difference.
    """

    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])


###############################################################################
def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute metric for multiple paths.

    Notes
    -----
    Assume nodes in ground truth and proposed graph have the same names.
    Assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate
    https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/shortest_paths.html

    Parameters
    ----------
    all_pairs_lengths_gt : dict
        Dictionary of path lengths for ground truth graph.
    all_pairs_lengths_prop : dict
        Dictionary of path lengths for proposal graph.
    control_nodes : list
        List of control nodes to evaluate.
    min_path_length : float
        Minimum path length to evaluate.
    diff_max : float
        Maximum value to return. Defaults to ``1``.
    missing_path_len : float
        Value to assign a missing path.  Defaults to ``-1``.
    normalize : boolean
        Switch to normalize outputs. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    C, diffs, routes, diff_dic
        C is the APLS score
        diffs is a list of the the route differences
        routes is a list of routes
        diff_dic is a dictionary of path differences
    """

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()

    print()
    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    if verbose:
        print("\nComputing path_sim_metric()...")
        print("good_nodes:", good_nodes)

    # iterate overall start nodes
    # for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if verbose:
            print("start node:", start_node)
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            if verbose:
                print("for ss, node", start_node, "not in set")
                print("   skipping N paths:", len(
                    list(all_pairs_lengths_prop[start_node].keys())))
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            if verbose:
                print ("start_node missing:", start_node)
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            # end_nodes_gt_set = set(paths.keys()) # old version with all nodes

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print("missing nodes:", missing_nodes)

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
                # for end_node, len_gt in paths.iteritems():

                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max
                    len_prop = missing_path_len

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

                if verbose:
                    print("start_node={}, end_node={}".format(start_node, end_node))
                    print("   len_gt={:0.1f}, len_prop={:0.2f}, diff={:0.2f}".format(len_gt, len_prop, diff))

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        print("Return, no good_nodes")
        return 0, [], [], {}

    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot
    if verbose:
        print("Time to compute metric (score = ", C, ") for ", len(diffs),
              "routes:", time.time() - t0, "seconds")

    return C, diffs, routes, diff_dic


###############################################################################
def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir='', min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute APLS metric and plot results (optional)

    Notes
    -----
    Computes APLS and creates plots in res_dir (if it is not empty)

    Arguments
    ---------
    all_pairs_lengths_gt_native : dict
        Dict of paths for gt graph.
    all_pairs_lengths_prop_native : dict
        Dict of paths for prop graph.
    all_pairs_lengths_gt_prime : dict
        Dict of paths for gt graph with control points from prop.
    all_pairs_lengths_prop_prime : dict
        Dict of paths for prop graph with control points from gt.
    control_points_gt : list
        Array of control points.
    control_points_prop : list
        Array of control points.
    res_dir : str
        Output dir for plots.  Defaults to ``''`` (no plotting).
    min_path_length : float
        Minimum path length to evaluate.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the total APLS score
        C_gt_onto_prop is the score when inserting gt control nodes onto prop
        C_prop_onto_gt is the score when inserting prop control nodes onto gt
    """

    t0 = time.time()

    # return 0 if no paths
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        print("len(all_pairs_lengths_gt_native.keys()) == 0)")
        return 0, 0, 0

    ####################
    # compute metric (gt to prop)
    if verbose:
        print("Compute metric (gt snapped onto prop)")
    # control_nodes = all_pairs_lengths_gt_native.keys()
    control_nodes = [z[0] for z in control_points_gt]
    if verbose:
        print(("control_nodes_gt:", control_nodes))
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt1 = time.time() - t0
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))

    if len(res_dir) > 0:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        # can't plot route names if there are too many...
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]

#         apls_plots.plot_metric(
#             C_gt_onto_prop, diffs, routes_str=routes_str,
#             figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
#             scatter_png=scatter_png,
#             hist_png=hist_png)

    ######################

    ####################
    # compute metric (prop to gt)
    if verbose:
        print("Compute metric (prop snapped onto gt)")
    t1 = time.time()
    # control_nodes = all_pairs_lengths_prop_native.keys()
    control_nodes = [z[0] for z in control_points_prop]
    if verbose:
        print("control_nodes:", control_nodes)
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt2 = time.time() - t1
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))
    if len(res_dir) > 0:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
#         apls_plots.plot_metric(
#             C_prop_onto_gt, diffs, routes_str=routes_str,
#             figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
#             scatter_png=scatter_png,
#             hist_png=hist_png)

    ####################

    ####################
    # Total
    if verbose:
        print("C_gt_onto_prop, C_prop_onto_gt:", C_gt_onto_prop, C_prop_onto_gt)
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0
    if verbose:
        print("Total APLS Metric = Mean(", np.round(C_gt_onto_prop, 2), "+",
              np.round(C_prop_onto_gt, 2),
              ") =", np.round(C_tot, 2))
    if verbose:
        print("Total time to compute metric:", str(dt1 + dt2), "seconds")

    return C_tot, C_gt_onto_prop, C_prop_onto_gt
