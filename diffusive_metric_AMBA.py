# -*- coding: utf-8 -*-
"""
Diffusive distances at AMBA
@author: LABRA
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd



n = 41    # number of nodes / districts

# District's data (up to july 7 2020)
population = np.array([597969, 96701, 105552, 255073, 219031,
                      109695, 180914, 462827, 119805, 541691,
                      378167, 174883, 77161, 267655, 356392,
                      31023, 62921, 370900, 517082, 425265,
                      307443, 713947, 66466, 606413, 105918,
                      292224, 462998, 128096, 365771, 3075646,
                      61783, 36545, 17412, 193583, 2281194,
                      648312, 359953, 318632, 664783, 304122,
                      344067])
infected = np.array([1411, 92, 41, 358, 340, 106, 226, 1412,
                       89, 491, 433, 267, 67, 443, 1766, 16,
                       39, 676, 779, 1306, 417, 596, 79, 620,
                       200, 645, 677, 83, 708, 29142, 66, 14,
                       26, 356, 4033, 1680, 460, 580, 1941, 553,
                       1158])

## Node weights
a1 = np.ones(n)   # uniform        
a2 = infected/population   # density of the disease

## Affinities - Edge weights
# SUBE
df_sube = pd.read_excel("Matrix SUBE.xlsx", header=None)
A0p = df_sube.values
A0 = A0p/(sum(sum(A0p)))
# Neighbours
df_neighb = pd.read_excel("Matrix Neighbors.xlsx", header=None)
A1p = df_neighb.values
A1 = A1p/(sum(sum(A1p)))
# Length of the border between neighbour districts
df_frontier = pd.read_excel("Matrix Frontiers.xlsx", index_col=0)
A2p = df_frontier.values
A2 = A2p/(sum(sum(A2p)))
# Minimum of the populations between neighbours
unos = np.ones(n,dtype=np.byte)
pop1 = np.outer(unos,population)
pop_ady_min = np.minimum(pop1,pop1.T)
A4p = pop_ady_min * A1p
A4 = A4p/(sum(sum(A4p)))
# Length of the border * mimimum population between neighbours
A3p = pop_ady_min * A2p
A3 = A3p/(sum(sum(A3p)))

# Convex combinations
theta = 0.5
Af_1 = A0
Af_2 = A4
Amix = theta*Af_1 + (1-theta)*Af_2


###### Initialization of parameters and graph weights ######
t = 0.25

aa = a1   # vector of node weights
W = A0    # matrix of edge weights
######


## Laplacian

a = aa/sum(aa)  # node weights normalization
A = np.diag(a)  # diagonal matrix of node weights
Ai = np.linalg.inv(A)  # A^(-1)
a_rc = np.sqrt(a)
A_rc = np.diag(a_rc)  # A^(1/2)
ai_rc = 1/a_rc
Ai_rc = np.diag(ai_rc)  # A^(-1/2)

w_cols = W.sum(axis=0)  # Vector of sums of the elements of the columns of W
D = np.diag(w_cols)

L = Ai.dot(D-W)  # Laplacian

Ln = Ai_rc @ (D-W) @ Ai_rc  # Normalized laplacian


## Spectral análisis

val, vecLn = np.linalg.eigh(Ln)  # eigenvalues and eigenvectors of Ln
if abs(val[0])<0.0000001: val[0]=0  # 1st eigenvalue rounding error correction

# Eigenvalues of Ln and L coincide, but the eigenvectors of L result when
# pre-multiply the eigenvectors of Ln by A^(-1/2)
vecL = Ai_rc @ vecLn
# Columns of vecL constitute an A-ortonormal base of L^2(a)


### Heat kernel at time t

def heat_ker(t,val,vecL):
    expval = np.exp(-t*val)
    return (expval*vecL)@(vecL.T)
H = heat_ker(t,val,vecL)


### Diffusive distance at time t

def heat_dist(H,i,j):  # diffusive distance between two nodes
    return np.sqrt( (H[i]-H[j]) @ A @ (H[i]-H[j]) )

def matrix_dist(H):  # matrix of distances between nodes
    m_dist = np.zeros((n, n))
    for v in range(n):
        for w in range(n):
            m_dist[v,w] = heat_dist(H,v,w)
    return m_dist
M_dist = matrix_dist(H)

## Saving the matrix of distances in Excel file (choose file name accordingly):
# df_mdist = pd.DataFrame(M_dist)
# df_mdist.to_excel(f'Diffusive distance t={t} uniform SUBE.xlsx', index=False, header=None)


### Graph plots

G = nx.from_numpy_matrix(np.matrix(W))

labeldict = {}
for m in range(n):
    labeldict[m] = m+1

pos = { 0:(0.05,-1.3), 1:(0.78,-3), 2:(-1,3), 3:(-0.9,1.95), 4:(-0.55,-1.55),
5:(-1.39,0.05), 6:(-0.8,0.45), 7:(0.02,-0.6), 8:(-1.75,0.7), 9:(-1.1,0.5),
10:(-1.3,1.4), 11:(-0.3,1.5), 12:(-0.4,-2.55), 13:(-0.05,0.55), 14:(0.17,-0.3),
15:(0,-3.2), 16:(-0.9,-2.6), 17:(-0.25,-1.3), 18:(0.25,-1.9), 19:(-0.26,0.57),
20:(-1,1.3), 21:(0.55,-2.9), 22:(-1.1,-1.05), 23:(-0.9,-0.3), 24:(-0.05,-2),
25:(-0.15,1.02), 26:(-0.5,2.1), 27:(-1.7,3.1), 28:(0.45,-1.5), 29:(0,0),
30:(0.7,-2.4), 31:(-1.8,2.1), 32:(-1.5,-1.5), 33:(-0.56,0.75), 34:(-0.45,-0.65),
35:(-0.15,-0.8), 36:(-0.7,1.5), 37:(-0.6,0), 38:(0.3,-0.85), 39:(-0.78,1.05),
40:(-0.4,0.25) }


## Figure 1: graph

def color_edges(G):
    color_edg = []
    for i, j in G.edges:
        color_edg.append(W[i,j])
    return color_edg

def vminmax(M_dist, center):
    d_cent = M_dist[:,center]   # distances with the 'center' district
    d_cent_others = np.delete(d_cent, np.argmin(d_cent))
    r = np.max(d_cent)-np.min(d_cent_others)
    
    vmin = max(np.min(d_cent_others) - 0.1*r, 0)
    vmax = np.max(d_cent_others) + 0.1*r
    return vmin, vmax

def graph_draw(G, center, M_dist, vmin, vmax, edge_vmin, edge_vmax, color_edg):
    plt.figure()
    nx.draw(G, pos, node_color = M_dist[:,center], cmap = plt.cm.viridis_r,
            edge_vmin = edge_vmin, edge_vmax = edge_vmax,
            edge_color = color_edg, edge_cmap = plt.cm.Greys,
            vmin = vmin, vmax = vmax,
            labels=labeldict, font_size=10)
    scalarmap = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(scalarmap)
    plt.show()

center = 29   # center of the ball (CABA = 29)
color_edg = color_edges(G)
vmin, vmax = vminmax(M_dist,center)
graph_draw(G, center, M_dist, vmin, vmax, -0.025, 0.025, color_edg)
# plt.savefig(f'graph t={t} uniform SUBE.png', facecolor='ghostwhite')


## Figure 2: colored AMBA map

map_amba = gpd.read_file("amba.json")  # load of the map of AMBA districts
def map_draw(M_dist, vmin, vmax, center):
    map_amba['Distancia'] = M_dist[center]
    fig, ax = plt.subplots(figsize=(6,6))
    map_amba.plot(column='Distancia', ax=ax, cmap='viridis_r',
               vmin=vmin, vmax=vmax)
    ax.axis('off')
    scalarmap = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(scalarmap, fraction=0.045, pad=0.02)
    plt.tight_layout()

vmin, vmax = vminmax(M_dist,center)
map_draw(M_dist, vmin, vmax, center)
# plt.savefig(f'map of AMBA t={t} uniform SUBE.png')

