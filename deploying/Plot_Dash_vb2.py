# -------------------------------------------------------------------------------------------------------------------- #
# Dash Plot
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from scipy.interpolate import griddata
from plotly.subplots import make_subplots
import pandas as pd
# from fem import find_node_range

# -------------------------------------------------------------------------------------------------------------------- #
# 1 Data Import
# -------------------------------------------------------------------------------------------------------------------- #

from numpy import load
import pickle
import os

path_cwd = os.getcwd()
# path = os.path.join(path_cwd, 'data_out')
path = 'deploying\\data_out\\data_20260612_1747_casexx\\150000'

with open(os.path.join(path, 'mat_res_norm.pkl'),'rb') as handle:
		mat_res_pkl = pickle.load(handle)

# mat_res_raw0 = pd.DataFrame.from_dict(mat_res_pkl)
# mat_res_raw = mat_res_raw0.loc[0,:]
mat_res_raw = mat_res_pkl

BC = mat_res_raw['BC']
COORD = mat_res_raw['COORD']
ELEMENTS = mat_res_raw['ELEMENTS']
fe = mat_res_raw['fe']
GEOMA = mat_res_raw['GEOMA']
GEOMK = mat_res_raw['GEOMK']
MASK = mat_res_raw['MASK']
NODESG = mat_res_raw['NODESG']
POST = mat_res_raw['POST']
gauss_order = mat_res_raw['gauss_order']
na = mat_res_raw['na']
ux = mat_res_raw['ux'] 
uy = mat_res_raw['uy'] 
uz = mat_res_raw['uz']
thx = mat_res_raw['thx']
thy = mat_res_raw['thy']
thz = mat_res_raw['thz']
# print(mat_res_raw)


# -------------------------------------------------------------------------------------------------------------------- #
# 2 Textfile Export
# -------------------------------------------------------------------------------------------------------------------- #

# 2.0 Auxiliary Functions
def vect_for_txt_export(str1,str2,add_coord = True):
    for ia in range(na):
        col_t = POST[str2][ia]
        col_t = np.ndarray.reshape(col_t,len(col_t),1)
        if add_coord:
            coords_t = COORD[str1][1][ia]
            all_t = np.concatenate((coords_t,col_t),axis=1)
        else:
            all_t = col_t
        if ia == 0:
            all = all_t
        else:
            all = np.concatenate((all,all_t),axis=0)
    return all

# 2.1 Coordinates
temp = np.arange(1,len(NODESG[:,0])+1,dtype=int)
temp = np.ndarray.reshape(temp,len(temp),1)
Koordinaten = np.concatenate((temp,NODESG),axis=1)
np.savetxt(os.path.join(path, 'Koordinaten.txt'),Koordinaten)

# 2.2 Node Connectivity
temp = np.arange(1,len(ELEMENTS[:,0])+1,dtype=int)
temp = np.ndarray.reshape(temp,len(temp),1)
Knoten = np.concatenate((temp,ELEMENTS+1),axis=1)
np.savetxt(os.path.join(path, 'Knoten.txt'),Knoten)

# 2.3 Integration Point Results
# 2.3.1 Steel Stresses
str1 = 'ip'
str2 = 'ssxsupa'
ssxsup = vect_for_txt_export(str1,str2)
ssxsup = np.concatenate((ssxsup,np.zeros((len(ssxsup[:,0]),3))),axis=1)

str2 = 'ssysupa'
ssysup = vect_for_txt_export(str1,str2)
ssysup = np.concatenate((ssysup,np.zeros((len(ssysup[:,0]),3))),axis=1)

str2 = 'ssxinfa'
ssxinf = vect_for_txt_export(str1,str2)
ssxinf = np.concatenate((ssxinf,np.zeros((len(ssxinf[:,0]),3))),axis=1)

str2 = 'ssyinfa'
ssyinf = vect_for_txt_export(str1,str2)
ssyinf = np.concatenate((ssyinf,np.zeros((len(ssyinf[:,0]),3))),axis=1)

sig_s = np.concatenate((ssxinf,np.concatenate((ssyinf[:,[3,4,5,6]],np.concatenate((ssysup[:,[3,4,5,6]],ssxsup[:,[3,4,5,6]]),axis=1)),axis=1)),axis=1)
np.savetxt(os.path.join(path, 'sig_s.txt'),sig_s)

# 2.3.2 CFRP Stresses
str1 = 'ip'
str2 = 'spxsupa'
spxsup = vect_for_txt_export(str1,str2)
spxsup = np.concatenate((spxsup,np.zeros((len(spxsup[:,0]),3))),axis=1)

str2 = 'spysupa'
spysup = vect_for_txt_export(str1,str2)
spysup = np.concatenate((spysup,np.zeros((len(spysup[:,0]),3))),axis=1)

str2 = 'spxinfa'
spxinf = vect_for_txt_export(str1,str2)
spxinf = np.concatenate((spxinf,np.zeros((len(spxinf[:,0]),3))),axis=1)

str2 = 'spyinfa'
spyinf = vect_for_txt_export(str1,str2)
spyinf = np.concatenate((spyinf,np.zeros((len(spyinf[:,0]),3))),axis=1)

sig_p = np.concatenate((spxinf,np.concatenate((spyinf[:,[3,4,5,6]],np.concatenate((spysup[:,[3,4,5,6]],spxsup[:,[3,4,5,6]]),axis=1)),axis=1)),axis=1)
np.savetxt(os.path.join(path, 'sig_p.txt'),sig_p)

# 2.3 Center Point Results
# 2.3.1 Stress Resultants
str1 = 'c'
allstr = ['Mx','My','Mxy','Nx','Ny','Nxy','Qx','Qy']

for i in range(len(allstr)+1):
    if i < 7.5:
        str2 = allstr[i]
        MNVi = vect_for_txt_export(str1,str2,add_coord = False)
    else:
        str2 = 'Mx'
        MNVi = vect_for_txt_export(str1, str2, add_coord=True)
    if i == 0:
        MNV = MNVi
    else:
        MNV = np.concatenate((MNV,MNVi),axis=1)
np.savetxt(os.path.join(path, 'M_N_V.txt'),MNV)

# -------------------------------------------------------------------------------------------------------------------- #
# 3 Plot
# -------------------------------------------------------------------------------------------------------------------- #

# 3.0 Auxiliary Functions

def plot_nodes():
    """ ------------------------------------- Create Nodes for Dash Plot  -------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - ELEMENTS, COORD,u
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - nodes[Entire/Area][Undeformed,Deformed]: global node coordinates in undef. and def. state
        - elements[entire/area]: Element connectivity
        - [nx,ny,nz] = nodes[Entire][Undeformed]
        - Indicators_1 = [Entire, a1, a2,...,na]
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Initiate
    nodes = {}
    elements = {}
    nodes['Entire']={}
    elements['Entire']=ELEMENTS

    # 1 Iteration through areas
    # - numelsia: number of elements in area ia
    # - elements[ia]: node connectivity of elements in area ia, shape (numelsia, 4)
    for ia in range(na):
        nodes[str(ia)]={}
        numelsia = len(np.array(GEOMK["ak"])[np.where(np.array(GEOMK["ak"])==ia)])
        elements[str(ia)]=np.reshape(ELEMENTS[np.where(np.array(GEOMK["ak"])==ia),:],(numelsia,4))

    # 2 Assignment of node coordinates for entire structure
    # - nx, ny, nz: node coordinates of all nodes
    # - nodes[Entire][Undeformed]: [nx,ny,nz]
    nx=np.array(COORD['n'][0][:,0]).reshape(len(ux),1)
    ny=np.array(COORD['n'][0][:,1]).reshape(len(ux),1)
    nz=np.array(COORD['n'][0][:,2]).reshape(len(ux),1)
    nodes['Entire']['Undeformed'] = [nx,ny,nz]

    # 3 Scale factor for plot scaling of deformations
    # - f: Scale factor
    # - nodes[Entire][Deformed] = [nx,ny,nz]+f*[ux,uy,uz]
    # if np.max(abs(nz))>0:
    #     f = np.max(abs(nz))/np.max(abs(uz))
    # else:
    #     f = np.max(abs(ny))/np.max(abs(uz))
    f = max(np.max(abs(nx)),np.max(abs(ny)),np.max(abs(nz)))/max(np.max(abs(ux)),np.max(abs(uy)),np.max(abs(uz)))/5
    nodes['Entire']['Deformed'] = [nx+f*ux,ny+f*uy,nz+f*uz]

    # 4 Assignment of node coordinates for each area
    # - nodes[ia][splot] for splot = [Deformed,Undeformed]: global node coordinates for individual areas
    for ia in range(na):
        nodes[str(ia)]['Undeformed'] = [nx[MASK[ia]], ny[MASK[ia]], nz[MASK[ia]]]
        nodes[str(ia)]['Deformed'] = [nx[MASK[ia]] + f * ux[MASK[ia]], ny[MASK[ia]] + f * uy[MASK[ia]],
                                      nz[MASK[ia]] + f * uz[MASK[ia]]]
        Indicators1 = list.append(indicators1, str(ia))
    return nodes, elements, nx, ny, nz, Indicators1


def plot_meshes():
    """ --------------------------------- Create Node Connectivity and Coordinates  ---------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - nodes, elements from plot_nodes() function
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - [xmesh, ymesh, zmesh] at positions [aplot (area),splot (def/undef)]: Mesh Points in correct order for
          plotting of elements: xmesh = [xi_e0, xi_e1,....xi_ene]
    -----------------------------------------------------------------------------------------------------------------"""

    # 0 Initiate
    xmesh = {}
    ymesh = {}
    zmesh = {}

    # 1 Iteration through areas
    for aplot in indicators1:

        # 1.0 Initiation of meshpoint vectors
        xmesh[str(aplot)]= {}
        ymesh[str(aplot)] = {}
        zmesh[str(aplot)] = {}

        # 1.1 Iteration through splot = [Deformed,Undeformed]
        for splot in indicators2:

            # 1.1.1 NODES_plt = coordinates of nodes to be plotted
            #       ELEMENTS_plt = node connectivity of plotted elements
            NODES_plt = nodes['Entire'][str(splot)]
            ELEMENTS_plt = elements[str(aplot)]

            # 1.1.2 Number of elements
            num_elements = len(ELEMENTS_plt[:, 0])

            # 1.1.3 Initiation of xall,yall,zall: vectors of plotted coordinates for nodes in correct element order
            xall = np.array([])
            yall = np.array([])
            zall = np.array([])

            # 1.1.4 Iteration through elements of area aplot
            for k in range(num_elements):

                # 1.1.4.1 Nodes constituting regarded element
                nk = ELEMENTS_plt[k, :][ELEMENTS_plt[k, :]<10**5]

                # 1.1.4.2 Coordinates of nodes nk
                nodes_x = NODES_plt[0][nk]
                nodes_y = NODES_plt[1][nk]
                nodes_z = NODES_plt[2][nk]

                # 1.1.4.3 Append first node to end to plot edge n4-n1 (n3-n1 for tris) as well
                nodes_x = np.append(nodes_x, nodes_x[0])
                nodes_x = np.append(nodes_x, None)
                nodes_y = np.append(nodes_y, nodes_y[0])
                nodes_y = np.append(nodes_y, None)
                nodes_z = np.append(nodes_z, nodes_z[0])
                nodes_z = np.append(nodes_z, None)

                # 1.1.4.4 Assemble nodes_x/y/z of regarded element to all-vectors containing all points for regarded
                #         area
                xall = np.append(xall,nodes_x)
                yall = np.append(yall,nodes_y)
                zall = np.append(zall,nodes_z)

            # 1.1.5 Assign x/y/zall to x/y/zmesh lists with options aplot = [Entire/0/1/2/../na] and
            #       splot = [Deformed/Undeformed]
            xmesh[str(aplot)][str(splot)] = xall
            ymesh[str(aplot)][str(splot)] = yall
            zmesh[str(aplot)][str(splot)] = zall
            # print(xmesh)
    return xmesh,ymesh,zmesh


def plot_colors():
    """ ----------------------------------- Create Information for Colorplots  --------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - COORD, MASK, Solution output
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - colors[ia][c] = [x_ip,y_ip,c], c = value for searched parameter at integration/center point in local coord
    -----------------------------------------------------------------------------------------------------------------"""

    # 0 Initiate
    colors = {}

    # 1 Iteration throuhg areas
    for ia in range(na):

        # 1.0 Initiate colors
        colors[str(ia)]={}

        # 1.1 Local x- and y-coordinates of nodes of area ia
        xa = np.array(COORD['n'][2][ia][:, 0]).reshape(len(ux), 1)[MASK[ia]].flatten()
        ya = np.array(COORD['n'][2][ia][:, 1]).reshape(len(ux), 1)[MASK[ia]].flatten()

        # 1.2 Create colors-entries for Deformations and relative error of deformations (in nodes)
        for i in range(12):
            v=[ux, uy, uz,thx,thy,thz,POST["relunx"][0],POST["reluny"][0],POST["relunz"][0],POST["relthnx"][0],POST["relthny"][0],POST["relthnz"][0]][i]
            titles = ['ux','uy','uz','thx','thy','thz','relunx','reluny','relunz','relthnx','relthny','relthnz']
            colors[str(ia)][titles[i]] = colorgrids(xa,ya,v[MASK[ia]].flatten(),ia)

        # 1.3 Entries for residual forces (in element center points)
        colors[str(ia)]['RNx'] = colorgrids(xa, ya, POST['RNx'][1][ia], ia)
        colors[str(ia)]['RNy'] = colorgrids(xa, ya, POST['RNy'][1][ia], ia)
        colors[str(ia)]['RNxy'] = colorgrids(xa, ya, POST['RNxy'][1][ia], ia)

        # 1.4 Entries for residual moments (in element center points)
        colors[str(ia)]['RMx'] = colorgrids(xa, ya, POST['RMx'][1][ia], ia)
        colors[str(ia)]['RMy'] = colorgrids(xa, ya, POST['RMy'][1][ia], ia)
        colors[str(ia)]['RMxy'] = colorgrids(xa, ya, POST['RMxy'][1][ia], ia)

        # 1.5 Entries for moments (in element center points)
        colors[str(ia)]['Mx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Mx'][ia],ia)
        colors[str(ia)]['My'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['My'][ia], ia)
        colors[str(ia)]['Mxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Mxy'][ia], ia)

        # 1.6 Entries for membrane and shear forces (in element center points)
        colors[str(ia)]['Nx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Nx'][ia],ia)
        colors[str(ia)]['Ny'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Ny'][ia], ia)
        colors[str(ia)]['Nxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Nxy'][ia], ia)
        colors[str(ia)]['Qx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Qx'][ia],ia)
        colors[str(ia)]['Qy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Qy'][ia], ia)

        # 1.7 Entries for strains and steel stresses (in integration points) in top and bottom layer
        colors[str(ia)]['ex'] = {}
        colors[str(ia)]['ex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exsupa'][ia], ia)
        colors[str(ia)]['ex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exinfa'][ia], ia)
        colors[str(ia)]['ey'] = {}
        colors[str(ia)]['ey']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eysupa'][ia], ia)
        colors[str(ia)]['ey']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eyinfa'][ia], ia)
        colors[str(ia)]['gxy'] = {}
        colors[str(ia)]['gxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxysupa'][ia], ia)
        colors[str(ia)]['gxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxyinfa'][ia], ia)
        colors[str(ia)]['e3'] = {}
        colors[str(ia)]['e3']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3supa'][ia], ia)
        colors[str(ia)]['e3']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3infa'][ia], ia)
        colors[str(ia)]['e1'] = {}
        colors[str(ia)]['e1']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1supa'][ia], ia)
        colors[str(ia)]['e1']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1infa'][ia], ia)
        colors[str(ia)]['th'] = {}
        colors[str(ia)]['th']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thsupa'][ia], ia)
        colors[str(ia)]['th']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thinfa'][ia], ia)
        colors[str(ia)]['ssx'] = {}
        colors[str(ia)]['ssx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxsupa'][ia], ia)
        colors[str(ia)]['ssx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxinfa'][ia], ia)
        colors[str(ia)]['ssy'] = {}
        colors[str(ia)]['ssy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssysupa'][ia], ia)
        colors[str(ia)]['ssy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssyinfa'][ia], ia)
        colors[str(ia)]['spx'] = {}
        colors[str(ia)]['spx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxsupa'][ia], ia)
        colors[str(ia)]['spx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxinfa'][ia], ia)
        colors[str(ia)]['spy'] = {}
        colors[str(ia)]['spy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spysupa'][ia], ia)
        colors[str(ia)]['spy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spyinfa'][ia], ia)

        # 1.8 Entries for relative strain errors (in integration points) in top and bottom layer
        for key in ['relexsupa', 'relexinfa', 'releysupa', 'releyinfa', 'relgxysupa', 'relgxyinfa']: 
            POST[key] = np.nan_to_num(POST[key], nan = 0)
        colors[str(ia)]['relex'] = {}
        colors[str(ia)]['relex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexsupa'][ia], ia)
        colors[str(ia)]['relex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexinfa'][ia], ia)
        colors[str(ia)]['reley'] = {}
        colors[str(ia)]['reley']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releysupa'][ia], ia)
        colors[str(ia)]['reley']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releyinfa'][ia], ia)
        colors[str(ia)]['relgxy'] = {}
        colors[str(ia)]['relgxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxysupa'][ia], ia)
        colors[str(ia)]['relgxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxyinfa'][ia], ia)
    return colors


def colorgrids(x,y,c,ia):
    """ ---------------------------------------- Create Colorgrids  -------------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - x,y: Local coordinates of regarded points
        - c: Color at given coordinates x&y
        - ia: area number
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - xgrid,ygrid: local grid points at which colorgrid is defined
        - Ci: Color values at coordinates x/ygrid
    -----------------------------------------------------------------------------------------------------------------"""

    # 1 Local node coordinates of area ia
    nxL = np.array(COORD['n'][2][ia][:, 0]).reshape(len(ux), 1)[MASK[ia]].flatten()
    nyL = np.array(COORD['n'][2][ia][:, 1]).reshape(len(ux), 1)[MASK[ia]].flatten()

    # 1.1 minimum and maximum coordinates
    xmin = min(nxL)
    ymin = min(nyL)
    xmax = max(nxL)
    ymax = max(nyL)

    # 2 Create Grid
    # 2.1 Grid Spacing
    dmesh = GEOMA["meshsa"][ia][0]

    # 2.2 Creation of meshgrid
    xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / dmesh) * gauss_order)
    ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / dmesh) * gauss_order)
    if xgrid.size == 0 or ygrid.size == 0:
        xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / (dmesh/2)) * gauss_order)
        ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / (dmesh/2)) * gauss_order)
    X, Y = np.meshgrid(xgrid, ygrid)

    # 2.3 Assign color data to grid

    # # 2.3.1 Extrapolate Data (on nodes or integration points) to nodes by using method 'nearest'
    # cgrid = griddata((x, y), c, (nxL, nyL), method='nearest')

    # # 2.3.1 Interpolate previously extrapolated data to grid points for plotting. Using 'cubic' allows for
    # #       arbitrary shapes to be contour filled
    # C = griddata((nxL, nyL), cgrid, (X, Y),method='cubic')
    # return [xgrid,ygrid,C]

    Xflat = X.flatten()
    Yflat = Y.flatten()

    from scipy.interpolate import Rbf
    rbf3 = Rbf(x, y, c, function="linear", smooth=1)
    cgrid = rbf3(Xflat,Yflat)
    return [Xflat,Yflat,cgrid]



def find_node_range(xmin, xmax, ymin, ymax, zmin, zmax):
    NODESG = COORD["n"][0]
    nodesx = NODESG[:,0]
    nodesy = NODESG[:,1]
    nodesz = NODESG[:,2]
    ind1 = np.array(np.where(nodesx<=xmax)).ravel()
    ind2 = np.array(np.where(nodesx>=xmin)).ravel()
    indx = np.intersect1d(ind1,ind2)

    ind1 = np.array(np.where(nodesy <= ymax)).ravel()
    ind2 = np.array(np.where(nodesy >= ymin)).ravel()
    indy = np.intersect1d(ind1, ind2)

    ind1 = np.array(np.where(nodesz <= zmax)).ravel()
    ind2 = np.array(np.where(nodesz >= zmin)).ravel()
    indz = np.intersect1d(ind1, ind2)

    indxy = np.intersect1d(indx,indy)
    ind = np.intersect1d(indxy,indz)

    return ind


# 3.1 Plot Options
"3.1 Output:    - Indicators: Options to choose for in dash plot"
"               - Titles: Titles for Color Plots"
"               - node and connectivity information from plot_nodes function"

indicators1 = ['Entire']
indicators2 = ['Undeformed','Deformed']
indicators3 = ['ux','uy','uz','thx','thy','thz','Mx','My','Mxy','Nx','Ny','Nxy','Qx','Qy','ex','ey','gxy','e3','e1','th','ssx','ssy',
               'spx','spy','RNx','RNy','RNxy','RMx','RMy','RMxy','relunx','reluny','relunz','relthnx','relthny','relthnz',
               'relex','reley','relgxy']
indicators4 = ['sup','inf']
Titles3 = {'ux':    '<i>u<sub>x</sub></i>',
           'uy':    '<i>u<sub>y</sub></i>',
           'uz':    '<i>u<sub>z</sub></i>',
           'thx':   '\u03B8<sub><i>x</i></sub>',
           'thy':   '\u03B8<sub><i>y</i></sub>',
           'thz':   '\u03B8<sub><i>z</i></sub>',
           'Mx':    '<i>M<sub>x</sub></i>',
           'My':    '<i>M<sub>y</sub></i>',
           'Mxy':   '<i>M<sub>xy</sub></i>',
           'Nx': '<i>N<sub>x</sub></i>',
           'Ny': '<i>N<sub>y</sub></i>',
           'Nxy': '<i>N<sub>xy</sub></i>',
           'Qx': '<i>Q<sub>x</sub></i>',
           'Qy': '<i>Q<sub>y</sub></i>',
           'ex':    '\u03B5<i><sub>x',
           'ey':    '\u03B5<i><sub>y',
           'gxy':    '\u03B3<i><sub>xy',
           'e3':    '\u03B5<i><sub>3',
           'e1':    '\u03B5<i><sub>1',
           'th':   '\u03B8',
           'ssx':    '\u03C3<i><sub>sx',
           'ssy':    '\u03C3<i><sub>sy',
           'spx': '\u03C3<i><sub>px',
           'spy': '\u03C3<i><sub>py',
           'RNx': '<i>RN<sub>x</sub></i>',
           'RNy': '<i>RN<sub>y</sub></i>',
           'RNxy': '<i>RN<sub>xy</sub></i>',
           'RMx': '<i>RM<sub><i>x</i></sub>',
           'RMy': '<i>RM<sub><i>y</i></sub>',
           'RMxy': '<i>RM<sub><i>xy</i></sub>',
           'relunx': '\u03B4<i>u<sub>x</sub></i>/<i>u<sub>x',
            'reluny': '\u03B4<i>u<sub>y</sub></i>/<i>u<sub>y',
            'relunz': '\u03B4<i>u<sub>z</sub></i>/<i>u<sub>z',
           'relthnx': '\u03B4\u03B8<i><sub>x</sub></i>/\u03B8<i><sub>x',
           'relthny': '\u03B4\u03B8<i><sub>y</sub></i>/\u03B8<i><sub>y',
           'relthnz': '\u03B4\u03B8<i><sub>z</sub></i>/\u03B8<i><sub>z',
            'relex':'\u03B4\u03B5<i><sub>x</sub></i>/\u03B5<i><sub>x',
            'reley':'\u03B4\u03B5<i><sub>y</sub></i>/\u03B5<i><sub>y',
            'relgxy':'\u03B4\u03B3<i><sub>xy</sub></i>/\u03B3<i><sub>xy',
           }

# 3.2 Nodes for Plotting and Element Connectivity
" 3.2 Output:   - nodes[Entire/Area][Undeformed,Deformed]: global node coordinates in undef. and def. state"
"               - elements[entire/area]: Element connectivity"
"               - [nx,ny,nz] = nodes[Entire][Undeformed]"
"               - Indicators_1 = [Entire, a1, a2,...,na]"
[nodes, elements, nx, ny, nz, Indicators1] = plot_nodes()

# 3.3 Mesh for Plots
" 3.3 Output:   - Mesh from plot_meshes() function: in function of aplot and splot: Node connectivity and coordinates"
[xmesh,ymesh,zmesh] = plot_meshes()

# 3.4 Color grids
" 3.4 Output:   - colors = [x_ip,y_ip,c], c = value for searched parameter at integration/center point in local coord."
colors = plot_colors()

# 3.5 Initiation of Dash Plot and Camera Settings
" 3.5 Output:   - fig,app,camera"
fig = make_subplots(rows=1,cols=2,specs=[[{'is_3d': True}, {'is_3d': False}]])
app = dash.Dash(__name__)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=-1.75, z=1.25)
)

# 3.6 Create App
" 3.6 Output:   app.layout with dropdown menus and checkboxes"
app.layout = html.Div([
    html.Div([
        html.Label('Shape',style={'font-size': '120%'}),
        dcc.Dropdown(
            id='splot',
            options=[{'label': i, 'value': i} for i in indicators2],
            value='Undeformed'
        ),
    ],
    style={'width': '10%', 'display': 'inline-block', 'vertical-align':'top','margin-left':'70px'}
    ),

    html.Div([
        html.Label('Plot Settings', style={'font-size': '120%'}),
        dcc.Checklist(
            id = 'varplot',
            options=[
                {'label': 'Boundaries', 'value': 'Boundaries'},
                {'label': 'Forces', 'value': 'Forces'},
                {'label': 'Node Numbers', 'value': 'Node Numbers'},
                {'label': 'Coord Sys', 'value': 'Coord Sys'}
                ],
            value=[]
        ),
    ],
        style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '70px'}
    ),

    html.Div([
        html.Label('Area',style={'font-size': '120%'}),
        dcc.Dropdown(
            id='aplot',
            options=[{'label': i, 'value': i} for i in indicators1[1:len(indicators1)]],
            value='0',
        ),
    ],
    style = {'width': '10%','display':'inline-block', 'vertical-align':'top','margin-left':'350px'}
    ),

    html.Div([
        html.Label('Layer', style={'font-size': '120%', 'vertical-align': 'top'}),
        dcc.Dropdown(
            id='lplot',
            options=[{'label': i, 'value': i} for i in indicators4],
            value='sup',
        )
    ],
        style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}
    ),

    html.Div([
        html.Label('Contour',style={'font-size': '120%'}),
        dcc.Dropdown(
            id='cplot',
            options=[{'label': i, 'value': i} for i in indicators3],
            value='ux',
        )
    ],
    style={'width': '10%','display':'inline-block', 'vertical-align':'top','margin-left':'20px'}
    ),

    dcc.Graph(id='FEM')
])

# 3.7 Update Graph & Callback
" 3.7 Output:   - Dash App"
"               - Callback for dropdowns and checkboxes"
@app.callback(
    Output('FEM', 'figure'),
    [Input('aplot', 'value'),
     Input('splot', 'value'),
     Input('cplot', 'value'),
     Input('lplot','value'),
     Input('varplot','value')])
def update_graph(aplot,splot,cplot,lplot,varplot):

    fig.data=[]

    # 4.9.1 Plot Deformed Shape
    fig.add_scatter3d(x=xmesh['Entire'][str(splot)],
                   y=ymesh['Entire'][str(splot)],
                   z=zmesh['Entire'][str(splot)],
                   mode='lines',
                   row=1,col=1,
                   showlegend = False)

    # 4.9.2 Plot Undeformed Shape
    fig.add_scatter3d(x=xmesh['Entire']['Undeformed'],
                   y=ymesh['Entire']['Undeformed'],
                   z=zmesh['Entire']['Undeformed'],
                   mode='lines',
                   line = dict(color = 'rgb(100,100,100)',width = 0.5),
                   row=1,col=1,
                   showlegend = False)

    # 4.9.3 Mark Active Area
    fig.add_scatter3d(x=xmesh[str(aplot)][str(splot)],
                   y=ymesh[str(aplot)][str(splot)],
                   z=zmesh[str(aplot)][str(splot)],
                   mode='lines',
                   line = dict(color = 'rgb(0,200,0)',width = 1),
                   surfacecolor = 'rgb(0,200,0)',
                   row=1,col=1,
                   showlegend = False)


    # 4.9.3 Plot Boundaries
    def plot_boundaries():
        mesh_size_plt = GEOMA["meshsa"][0][0]
        # B_plt = GEOMA["Ba"][0]
        B_plt = 1600
        order = 1
        lw = 5 * np.max(mesh_size_plt) / B_plt / order

        def plot_triangle(coord, d, i):
            d = d / order
            xtr = [coord[0]-d/2,coord[0]-d/2,coord[0]+d/2,coord[0]+d/2,coord[0]-d/2,coord[0],coord[0]+d/2,coord[0]+d/2,coord[0],coord[0]-d/2]
            ytr = [coord[1]+d/2,coord[1]-d/2,coord[1]-d/2,coord[1]+d/2,coord[1]+d/2,coord[1],coord[1]+d/2,coord[1]-d/2,coord[1],coord[1]-d/2]
            ztr = [coord[2]-d,coord[2]-d,coord[2]-d,coord[2]-d,coord[2]-d,coord[2],coord[2]-d,coord[2]-d,coord[2],coord[2]-d]
            fig.add_scatter3d(x=xtr,
                              y=ytr,
                              z=ztr,
                              mode='lines',
                              line=dict(color='rgb(0,0,0)', width=lw),
                              row=1, col=1,
                              showlegend=False)
            if BC[i, 6] == 1:
                fig.add_cone(x=[coord[0]+d/3], y=[coord[1]], z=[coord[2]], u=[-d], v=[0], w=[0], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                             showlegend=False)
            if BC[i, 7] == 1:
                fig.add_cone(x=[coord[0]], y=[coord[1]+d/3], z=[coord[2]], u=[0], v=[-d], w=[0], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                             showlegend=False)
            if BC[i, 8] == 1:
                fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]+d/3], u=[0], v=[0], w=[-d], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                             showlegend=False)
            if BC[i, 9] == 1:
                fig.add_cone(x=[coord[0]-d/3], y=[coord[1]], z=[coord[2]], u=[2*d/3], v=[0], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)
                fig.add_cone(x=[coord[0]-d/2], y=[coord[1]], z=[coord[2]], u=[2*d/3], v=[0], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)
            if BC[i, 10] == 1:
                fig.add_cone(x=[coord[0]], y=[coord[1]-d/3], z=[coord[2]], u=[0], v=[2*d/3], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)
                fig.add_cone(x=[coord[0]], y=[coord[1]-d/2], z=[coord[2]], u=[0], v=[2*d/3], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)
            if BC[i, 11] == 1:
                fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]-d/3], u=[0], v=[0], w=[2*d/3], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)
                fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]-d/2], u=[0], v=[0], w=[2*d/3], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                             showlegend=False)

        """--------------------------------------------------------------------------------------------------------------"""
        numb = len(BC[:, 0])
        for i in range(numb):
            if BC[i, 6] + BC[i, 7] + BC[i, 8] + BC[i, 9] + BC[i, 10] > 0:
                xmin = BC[i, 0]
                xmax = BC[i, 1]
                ymin = BC[i, 2]
                ymax = BC[i, 3]
                zmin = BC[i, 4]
                zmax = BC[i, 5]
                nodesi = find_node_range(xmin, xmax, ymin, ymax, zmin, zmax)
                for j in nodesi:
                    j = int(j)
                    coord = NODESG[j, :]
                    plot_triangle(coord, mesh_size_plt / 2, i)

        """--------------------------------------------------------------------------------------------------------------"""
    if 'Boundaries' in varplot:
        plot_boundaries()
    # 4.9.4 Plot Forces
    def plot_forces():
        f_max = max(abs(fe))
        max_conesize = GEOMA["meshsa"][0][0]*3
        max_linesize = int(GEOMA["meshsa"][0][0])*3
        for i in range(len(fe)):
            fe_i = fe[i]
            if abs(fe_i) > 0.01:
                n_i = int(i/6)
                [x,y,z] = COORD["n"][0][n_i]
                dir = i - n_i*6
                factor = max(1/4,abs(fe_i/f_max))*np.sign(fe_i)
                if dir == 0 or dir == 3:
                    " Force in x - direction "
                    x = x - int(max_conesize * factor) / 3
                    u = max_conesize*factor
                    v = [0]
                    w = [0]

                    xl = [x,x-int(max_linesize*factor)*2/3]
                    yl = [y,y]
                    zl = [z,z]
                elif dir == 1 or dir == 4:
                    " Force in y - direction "
                    y = y - int(max_conesize * factor) / 3
                    u = [0]
                    v = max_conesize * factor
                    w = [0]

                    xl = [x, x]
                    yl = [y, y - int(max_linesize * factor)*2/3]
                    zl = [z, z]
                elif dir == 2 or dir == 5:
                    " Force in z - direction "
                    z = z - int(max_conesize * factor)/3
                    u = [0]
                    v = [0]
                    w = max_conesize * factor

                    xl = [x, x]
                    yl = [y, y]
                    zl = [z, z - int(max_linesize * factor)*2/3]
                if abs(factor) < 0.251:
                    if dir < 2.5:
                        color = 'orangered'
                    else:
                        color = 'green'
                else:
                    if dir < 2.5:
                        color = 'red'
                    else:
                        color = 'blue'
                fig.add_cone(x=[x], y=[y], z=[z], u=u, v=v, w=w, colorscale=[[0, color], [1,color]], showscale = False, showlegend=False)
                fig.add_scatter3d(x=xl,
                                  y=yl,
                                  z=zl,
                                  mode='lines',
                                  line=dict(color='rgb(255,0,0)',width=abs(int(max_conesize * factor))/100),
                                  row=1, col=1,
                                  showlegend=False)
    if 'Forces' in varplot:
        plot_forces()
    # 4.9.5 Plot Node Numbers
    if 'Node Numbers' in varplot:
        nn = len(NODESG[:, 0])
        for n in range(nn):
            fig.add_scatter3d(x=[NODESG[n][0]],
                              y=[NODESG[n][1]],
                              z=[NODESG[n][2]],
                              mode="markers+text",
                              text=str(n),
                              row=1, col=1,
                              showlegend=False)
        fig.update_traces(textposition='top center')
    def plot_coordsys():
        max_conesize = GEOMA["meshsa"][0][0]/2
        max_linesize = int(GEOMA["meshsa"][0][0])/2
        fig.add_scatter3d(x=[0,0],
                          y=[0,0],
                          z=[0,max_linesize],
                          mode='lines',
                          line=dict(color='rgb(0,0,255)', width=1),
                          row=1, col=1,
                          showlegend=False)
        fig.add_cone(x=[0], y=[0], z=[max_linesize], u=[0], v=[0], w=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                     showlegend=False)
        fig.add_scatter3d(x=[0],
                          y=[0],
                          z=[max_linesize*6/5],
                          mode="text",
                          textfont = dict(family = "times", color = "blue"),
                          text="<i>z</i>",
                          row=1, col=1,
                          showlegend=False)
        fig.add_scatter3d(x=[0,0],
                          z=[0,0],
                          y=[0,max_linesize],
                          mode='lines',
                          line=dict(color='rgb(0,0,255)', width=1),
                          row=1, col=1,
                          showlegend=False)
        fig.add_cone(x=[0], z=[0], y=[max_linesize], u=[0], w=[0], v=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                     showlegend=False)
        fig.add_scatter3d(x=[0],
                          z=[0],
                          y=[max_linesize*6/5],
                          mode="text",
                          textfont = dict(family = "times", color = "blue"),
                          text="<i>y</i>",
                          row=1, col=1,
                          showlegend=False)
        fig.add_scatter3d(y=[0,0],
                          z=[0,0],
                          x=[0,max_linesize],
                          mode='lines',
                          line=dict(color='rgb(0,0,255)', width=1),
                          row=1, col=1,
                          showlegend=False)
        fig.add_cone(y=[0], z=[0], x=[max_linesize], v=[0], w=[0], u=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                     showlegend=False)
        fig.add_scatter3d(y=[0],
                          z=[0],
                          x=[max_linesize*6/5],
                          mode="text",
                          textfont = dict(family = "times", color = "blue"),
                          text="<i>x</i>",
                          row=1, col=1,
                          showlegend=False)

        coordloc = np.array([[1,0,0],[0,1,0],[0,0,1]])@GEOMA["T1"][int(aplot)]
        Oo = GEOMA["Oa"][int(aplot)]
        for i in [0,1]:
            strr = ["<i>x'</i>","<i>y'</i>"][i]
            xx = [Oo[0] + coordloc[i][0] * max_linesize / 2]
            yy=  [Oo[1] + coordloc[i][1] * max_linesize / 2]
            zz = [Oo[2] + coordloc[i][2] * max_linesize / 2]
            uu = [int(coordloc[i][0]) * max_conesize]
            vv = [int(coordloc[i][1]) * max_conesize]
            ww = [int(coordloc[i][2]) * max_conesize]
            fig.add_scatter3d(x=[Oo[0],Oo[0] + coordloc[i][0] * max_linesize / 2],
                              y=[Oo[1],Oo[1] + coordloc[i][1] * max_linesize / 2],
                              z=[Oo[2],Oo[2] + coordloc[i][2] * max_linesize / 2],
                              mode='lines',
                              line=dict(color='rgb(0,200,0)', width=1),
                              row=1, col=1,
                              showlegend=False)
            fig.add_cone(x=xx,
                         y=yy,
                         z=zz,
                         u=uu,
                         v=vv,
                         w=ww,
                         colorscale=[[0, 'rgb(0,200,0)'], [1, 'rgb(0,200,0)']], showscale=False,
                         showlegend=False)
            fig.add_scatter3d(x=xx,
                              y=yy,
                              z=zz,
                              mode="text",
                              textfont = dict(family = "times", color = 'rgb(0,200,0)'),
                              text=strr,
                              row=1, col=1,
                              showlegend=False)

    if 'Coord Sys' in varplot:
        plot_coordsys()

    # 4.9.6 Colorplots
    if cplot == 'ex' or cplot == 'ey' or cplot == 'gxy' or cplot == 'e3' or cplot == 'e1' or cplot == 'th' or cplot == 'ssx' or cplot == 'ssy' or cplot == 'spx' or cplot == 'spy' or cplot == 'relex' or cplot == 'reley' or cplot == 'relgxy':
        xacl = colors[str(aplot)][str(cplot)][str(lplot)][0]
        yacl = colors[str(aplot)][str(cplot)][str(lplot)][1]
        zacl = colors[str(aplot)][str(cplot)][str(lplot)][2]
    else:
        xacl = colors[str(aplot)][str(cplot)][0]
        yacl = colors[str(aplot)][str(cplot)][1]
        zacl = colors[str(aplot)][str(cplot)][2]
    print(np.nanmin(zacl))
    fig.add_contour(x=xacl ,
                    y=yacl ,
                    z=zacl ,
                    contours_showlines=True,
                    # contours_coloring = 'heatmap',
                    colorscale='Spectral',row=1,col=2,
                    # contours=dict(start=np.nanmin(zacl),end=np.nanmax(zacl),size=(np.nanmax(zacl)-np.nanmin(zacl))/10),
                    colorbar=dict(tickfont=dict(family="Times New Roman"))
                    )
    # 4.9.7 Update
    fig.update_layout(scene=dict(
        xaxis=dict(showticklabels=False, showgrid=False, visible=False),
        yaxis=dict(showticklabels=False, showgrid=False, visible=False),
        zaxis=dict(showticklabels=False, showgrid=False, visible=False),
        camera = camera,
        aspectmode='data',
        ),
        xaxis = dict(showticklabels=False,showgrid=False, visible=False),
        yaxis = dict(showticklabels=False,showgrid=False, visible=False, scaleanchor = 'x', scaleratio = 1),
        annotations = [{'text':str(splot),'x':0,'y':1.1,'xref':'paper','yref':'paper','showarrow':False,'font':dict(size=16,family="times")},
                       {'text': Titles3[str(cplot)], 'x': 0.56, 'y': 1.1, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': dict(size=16,family="times")},
                       {'text': "max = %.2e" %(np.max(zacl)), 'x': 0.56, 'y': 0.08, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': dict(size=16,family="times")},
                       {'text': "min = %.2e" % (np.min(zacl)), 'x': 0.56, 'y': 0.0,
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': dict(size=16,family="times")}
                       ]
    )
    fig.data[0].line.color = 'rgb(0,0,0)'
    fig.update_xaxes()
    fig.update_yaxes()
    fig.layout.plot_bgcolor = '#fff'
    return fig

# 3.8 Run App on Server
# app.run_server(debug=False)
app.run(debug=False)
