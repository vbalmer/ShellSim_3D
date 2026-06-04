import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "text.latex.preamble": r'\usepackage{wasysym}'
})

def read(mat_res_raw):

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
    eps_g = mat_res_raw['eps_g']

    return BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g


def plot_nodes(mat_res_raw):
    """ ------------------------------------- Create Nodes for Dash Plot  -------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - ELEMENTS, COORD,u
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - nodes[Entire/Area][Undeformed,Deformed]: global node coordinates in undef. and def. state
        - elements[entire/area]: Element connectivity
        - [nx,ny,nz] = nodes[Entire][Undeformed]
        - Indicators_1 = [Entire, a1, a2,...,na]
    -----------------------------------------------------------------------------------------------------------------"""

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)


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
        indicators1 = ['Entire']
        Indicators1 = list.append(indicators1, str(ia))
    return nodes, elements, nx, ny, nz, Indicators1



def plot_meshes(mat_res_raw):
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
    indicators1 = [str(0)]
    for aplot in indicators1:

        # 1.0 Initiation of meshpoint vectors
        xmesh[str(aplot)]= {}
        ymesh[str(aplot)] = {}
        zmesh[str(aplot)] = {}

        # 1.1 Iteration through splot = [Deformed,Undeformed]
        indicators2 = ['Undeformed','Deformed']
        nodes, elements, nx, ny, nz, Indicators1 = plot_nodes(mat_res_raw)
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
    return xmesh,ymesh,zmesh


def colorgrids(x,y,c,ia, mat_res_raw):
    """ ---------------------------------------- Create Colorgrids  -------------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - x,y: Local coordinates of regarded points
        - c: Color at given coordinates x&y
        - ia: area number
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - xgrid,ygrid: local grid points at which colorgrid is defined
        - Ci: Color values at coordinates x/ygrid
    -----------------------------------------------------------------------------------------------------------------"""
    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

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
    meshgrid_scale = 2          # used to be meshgrid_scale = gauss_order
    xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / dmesh) * meshgrid_scale)
    ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / dmesh) * meshgrid_scale)
    if xgrid.size == 0 or ygrid.size == 0:
        xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / (dmesh/2)) * meshgrid_scale)
        ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / (dmesh/2)) * meshgrid_scale)
    X, Y = np.meshgrid(xgrid, ygrid)

    # 2.3 Assign color data to grid

    # # 2.3.1 Extrapolate Data (on nodes or integration points) to nodes by using method 'nearest'
    # cgrid = griddata((x, y), c, (nxL, nyL), method='nearest')

    # # 2.3.1 Interpolate previously extrapolated data to grid points for plotting. Using 'cubic' allows for
    # #       arbitrary shapes to be contour filled
    # C = griddata((nxL, nyL), cgrid, (X, Y),method='cubic')


    # return [xgrid,ygrid,C]
    Xflat = X   #.flatten()
    Yflat = Y   #.flatten()

    from scipy.interpolate import Rbf
    rbf3 = Rbf(x, y, c, function="linear", smooth=1)
    cgrid = rbf3(Xflat,Yflat)
    return [Xflat,Yflat,cgrid]



def plot_colors(mat_res_raw):
    """ ----------------------------------- Create Information for Colorplots  --------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - COORD, MASK, Solution output
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - colors[ia][c] = [x_ip,y_ip,c], c = value for searched parameter at integration/center point in local coord
    -----------------------------------------------------------------------------------------------------------------"""

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)


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
            colors[str(ia)][titles[i]] = colorgrids(xa,ya,v[MASK[ia]].flatten(),ia, mat_res_raw)

        # 1.3 Entries for residual forces (in element center points)
        colors[str(ia)]['RNx'] = colorgrids(xa, ya, POST['RNx'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RNy'] = colorgrids(xa, ya, POST['RNy'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RNxy'] = colorgrids(xa, ya, POST['RNxy'][1][ia], ia, mat_res_raw)

        # 1.4 Entries for residual moments (in element center points)
        colors[str(ia)]['RMx'] = colorgrids(xa, ya, POST['RMx'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RMy'] = colorgrids(xa, ya, POST['RMy'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RMxy'] = colorgrids(xa, ya, POST['RMxy'][1][ia], ia, mat_res_raw)

        # 1.5 Entries for moments (in element center points)
        colors[str(ia)]['Mx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Mx'][ia],ia, mat_res_raw)
        colors[str(ia)]['My'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['My'][ia], ia, mat_res_raw)
        colors[str(ia)]['Mxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Mxy'][ia], ia, mat_res_raw)

        # 1.6 Entries for membrane and shear forces (in element center points)
        colors[str(ia)]['Nx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Nx'][ia],ia, mat_res_raw)
        colors[str(ia)]['Ny'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Ny'][ia], ia, mat_res_raw)
        colors[str(ia)]['Nxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Nxy'][ia], ia, mat_res_raw)
        colors[str(ia)]['Qx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Qx'][ia],ia, mat_res_raw)
        colors[str(ia)]['Qy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Qy'][ia], ia, mat_res_raw)

        # 1.5 b Generalised Normal Strains
        
        colors[str(ia)]['epsx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 0].flatten(),ia, mat_res_raw)
        colors[str(ia)]['epsy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 1].flatten(),ia, mat_res_raw)
        colors[str(ia)]['epsxy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 2].flatten(),ia, mat_res_raw)

        # 1.6 b Generalised moment and shear strains

        colors[str(ia)]['chix'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 3].flatten(),ia, mat_res_raw)
        colors[str(ia)]['chiy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 4].flatten(),ia, mat_res_raw)
        colors[str(ia)]['chixy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 5].flatten(),ia, mat_res_raw)
        colors[str(ia)]['gamx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 6].flatten(),ia, mat_res_raw)
        colors[str(ia)]['gamy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 7].flatten(),ia, mat_res_raw)        

        # # 1.7 Entries for strains and steel stresses (in integration points) in top and bottom layer
        # colors[str(ia)]['ex'] = {}
        # colors[str(ia)]['ex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exsupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ey'] = {}
        # colors[str(ia)]['ey']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ey']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eyinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['gxy'] = {}
        # colors[str(ia)]['gxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['gxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxyinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['e3'] = {}
        # colors[str(ia)]['e3']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3supa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['e3']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3infa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['e1'] = {}
        # colors[str(ia)]['e1']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1supa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['e1']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1infa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['th'] = {}
        # colors[str(ia)]['th']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thsupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['th']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ssx'] = {}
        # colors[str(ia)]['ssx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxsupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ssx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ssy'] = {}
        # colors[str(ia)]['ssy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['ssy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssyinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['spx'] = {}
        # colors[str(ia)]['spx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxsupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['spx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['spy'] = {}
        # colors[str(ia)]['spy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['spy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spyinfa'][ia], ia, mat_res_raw)

        # # 1.8 Entries for relative strain errors (in integration points) in top and bottom layer
        # colors[str(ia)]['relex'] = {}
        # colors[str(ia)]['relex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexsupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['relex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['reley'] = {}
        # colors[str(ia)]['reley']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['reley']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releyinfa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['relgxy'] = {}
        # colors[str(ia)]['relgxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxysupa'][ia], ia, mat_res_raw)
        # colors[str(ia)]['relgxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxyinfa'][ia], ia, mat_res_raw)
    return colors


def find_node_range(xmin, xmax, ymin, ymax, zmin, zmax, mat_res_raw):

    COORD = mat_res_raw['COORD']
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


def plot_boundaries(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

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
        fig.scatter(x=xtr,
                    y=ytr,
                    z=ztr,
                    # mode='lines',
                    # line=dict(color='rgb(0,0,0)', width=lw),
                    # row=1, col=1,
                    # showlegend=False
                    )
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
        
        return

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
    return



def plot_forces(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

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
            # fig.add_cone(x=[x], y=[y], z=[z], 
            #              u=u, v=v, w=w, colorscale=[[0, color], [1,color]], showscale = False, showlegend=False)
            fig.scatter(xl,
                        yl,
                        zl,
                        # mode='lines',
                        # line=dict(color='rgb(255,0,0)',width=abs(int(max_conesize * factor))/100),
                        # row=1, col=1,
                        # showlegend=False
                        )


def plot_node_nr(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

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

def plot_coordsys(mat_res_raw, fig):
    
    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

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

    coordloc = np.array([[1,0,0],[0,1,0],[0,0,1]])@GEOMA["T1"][str(0)]
    Oo = GEOMA["Oa"][str(0)]
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



def main_plt(mat_res_raw:pd.DataFrame, id:str, path: str, nn = False, boundaries = True, forces = True, node_nr = True, coord_sys = True):
    """
    Plots 8 variables from output of simulation, for the single simulation specified in mat_res_raw.
    Note (units): The units here correspond to the ones of post-processing for sig, and u; for eps_g they correspond
                    to the eps_g in the matrix mat_res_raw (see definition of colors). 
                    The units have been checked to correspond to the simulation units [N, mm] everywhere.

    Input: 
    mat_res_raw     (pd.DataFrame)      One line of the data set that shall be investigated in more detail
    id              (str)               Identifier: "sig", "eps", "u", "geom+load"

    Output: 
    plt             (plt)               Plot of the desired 8 variables       

    """
    
    # Getting data
    [nodes, elements, nx, ny, nz, Indicators1] = plot_nodes(mat_res_raw)
    [xmesh,ymesh,zmesh] = plot_meshes(mat_res_raw)
    colors = plot_colors(mat_res_raw)
    
    
    # 0 - Plotting Geometry and Loads
    if id == 'geom+load':
        fig, ax = plt.subplots(1,2, subplot_kw={"projection": "3d"})

        xmesh_cl_deformed = [xi for xi in xmesh[str(0)]['Deformed'] if xi is not None]
        xmesh_cl_undeformed = [xi for xi in xmesh[str(0)]['Undeformed'] if xi is not None]
        ymesh_cl_deformed = [yi for yi in ymesh[str(0)]['Deformed'] if yi is not None]
        ymesh_cl_undeformed = [yi for yi in ymesh[str(0)]['Undeformed'] if yi is not None]
        zmesh_cl_deformed = [zi for zi in zmesh[str(0)]['Deformed'] if zi is not None]
        zmesh_cl_undeformed = [zi for zi in zmesh[str(0)]['Undeformed'] if zi is not None]

        plt.style.use('_mpl-gallery')

        # ax[0].plot_wireframe(xmesh_cl_undeformed,
        #             ymesh_cl_undeformed,
        #             zmesh_cl_undeformed, 
        #             rstride=10, cstride=10
        #             )

        # ax[1].plot_wireframe(xmesh_cl_deformed,
        #             ymesh_cl_deformed,
        #             zmesh_cl_deformed,
        #             #    mode='lines',
        #             #    line = dict(color = 'rgb(100,100,100)',width = 0.5),
        #             #    row=1,col=1,
        #             #    showlegend = False
        #             rstride=10, cstride=10
        #             )
        

        ax[0].scatter(xmesh_cl_undeformed,
                    ymesh_cl_undeformed,
                    zmesh_cl_undeformed, 
                    )

        ax[1].scatter(xmesh_cl_deformed,
                    ymesh_cl_deformed,
                    zmesh_cl_deformed,
                    )
    
        if boundaries:
            plot_boundaries(mat_res_raw, ax[0])
        if forces: 
            plot_forces(mat_res_raw, ax[0])
        # if node_nr:
        #     plot_node_nr(mat_res_raw, ax)
        # if coord_sys:
        #     plot_coordsys(mat_res_raw, ax)


    if nn:
        label_add = np.array([['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$']])
    elif not nn:
        label_add = np.array([['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$']])
   
    if id == 'sig':
        name = np.array([['Nx', 'Ny', 'Nxy'], 
                         ['Mx', 'My', 'Mxy'],
                         ['Qx', 'Qy', 'Qy']])
        labels_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'], 
                            ['$m_x$', '$m_y$', '$m_{xy}$'], 
                            ['$v_{xz}$', '$v_{yz}$', '$v_{yz}$']])
        labels = np.char.add(labels_sig, label_add)
        # label_colorbar = np.array(['$[kN/m]$', '$[kNm/m]$', '$[kN/m]$'])
        label_colorbar = np.array(['$[N/mm]$', '$[Nmm/mm]$', '$[N/mm]$'])
        n_rows = 3
        img = [0, 0, 0]
    
    elif id == 'eps':
        name = np.array([['epsx', 'epsy', 'epsxy'], 
                        ['chix', 'chiy', 'chixy'],
                        [ 'gamx', 'gamy', 'gamy']])
        labels_eps= np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'], 
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_{xz}$', r'$\gamma_{xz}$', r'$\gamma_{xz}$']])
        labels = np.char.add(labels_eps, label_add)
        label_colorbar = np.array([r'$[mm/mm]$', r'$[1/mm]$', r'$[mm/mm]$'])
        n_rows = 3
        img = [0, 0, 0]

        # # Convert [-] strains to [permille] strains
        # for i in range(3):
        #     for j in range(3):
        #         if i == 2 and j==2:
        #             pass
        #         else:
        #             colors[str(0)][name[i,j]][2] = colors[str(0)][name[i,j]][2]*10**(3)
    
    elif id == 'u':
        name = np.array([['ux','uy','uz'],
                          ['thx','thy','thz']])
        labels_u = np.array([['$u_x$','$u_y$','$u_z$'],
                         [r'$\vartheta_x$',r'$\vartheta_y$',r'$\vartheta_z$']])
        labels = np.char.add(labels_u, label_add[0:2, :])
        label_colorbar = np.array(['$[mm]$', '$[mrad]$'])
        n_rows = 2
        img = [0, 0]


    # Convert mm to m (for the x-/ y- axes)
    for i in range(n_rows):
        for j in range(3):
            if i == 2 and j==2:
                pass
            else:
                colors[str(0)][name[i,j]][0] = colors[str(0)][name[i,j]][0]*10**(-3)
                colors[str(0)][name[i,j]][1] = colors[str(0)][name[i,j]][1]*10**(-3)
    
    # Define max, min for colorbar
    min_temp, max_temp = np.zeros((name.shape)), np.zeros((name.shape))
    vmin, vmax = np.zeros((name.shape)), np.zeros((name.shape))
    for i in range(n_rows):
        for j in range(3):
            min_temp[i,j] = np.min(np.array(colors[str(0)][name[i,j]][2]))
            max_temp[i,j] = np.max(np.array(colors[str(0)][name[i,j]][2]))
        vmin[i,:] = np.min(min_temp[i,:])*np.ones((1,3))
        vmax[i,:] = np.max(max_temp[i,:])*np.ones((1,3))


    # Plot
    fig, axs = plt.subplots(n_rows,3, figsize = [n_rows*2.7, 8])
    for i in range(n_rows):
        for j in range(3):
            if i == 2 and j==2:
                axs[i,j].set_title(' ')
            else:
                axs[i,j].contour(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], colors[str(0)][name[i,j]][2],
                                        colors = 'black', linewidths = 0.5,
                                        vmin=vmin[i,j], vmax=vmax[i,j], 
                                        # levels = np.linspace(vmin[i,j], vmax[i,j], 10)
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                img[i] = axs[i,j].contourf(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], colors[str(0)][name[i,j]][2],
                                        vmin=vmin[i,j], vmax=vmax[i,j],
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                axs[i,j].set_title(labels[i,j]) 

    for i in range(n_rows):
        cbar = fig.colorbar(img[i], label = label_colorbar[i], ax = axs[i,0:3], location= 'right')
        # cbar.set_ticks(np.linspace(vmin[i,0], vmax[i,0], steps[i,0]))
        for label in cbar.ax.get_yticklabels()[::2]:
            label.set_visible(False)
    if n_rows == 3:
        axs[-1, -1].axis('off')
    for i in range(n_rows):
        for j in range(3):
            # axs[i,j] = plt.gca()
            axs[i,j].set_aspect('equal', 'box')
            axs[i,j].axis('square')
   
    
    # plt.tight_layout()
    if nn:
        plt.savefig(os.path.join(path, 'single_'+ id +'_NN'+'.png'), format = 'png')
        print(f'Saved single_{id}_NN.png')
    elif not nn:
        plt.savefig(os.path.join(path, 'single_'+ id +'_NLFEA'+'.png'), format = 'png')
        print(f'Saved single_{id}_NLFEA.png')
    # plt.show()
    # plt.close()


    return




def main_plt_delta(mat_res_raw:pd.DataFrame, mat_res_raw_NN: pd.DataFrame, id:str, path: str, type_err: str, stats: dict, comp_list = None, boundaries = True, forces = True, node_nr = True, coord_sys = True):
    """
    Plots 8 variables from output of simulation, for the single simulation specified in mat_res_raw.
    
    Input: 
    mat_res_raw     (pd.DataFrame)      One line of the data set that shall be investigated in more detail
    id              (str)               Identifier: "sig", "eps", "u", "geom+load"
    type_err        (str)               'max','rel','rRSE' or 'nRSE' (rel. max. error, rel. error, rel. RS error, normal. RS error)

    Output: 
    plt             (plt)               Plot of the desired 8 variables       

    """
    
    # Getting data
    [nodes, elements, nx, ny, nz, Indicators1] = plot_nodes(mat_res_raw)
    [xmesh,ymesh,zmesh] = plot_meshes(mat_res_raw)
    colors = plot_colors(mat_res_raw)
    colors_NN = plot_colors(mat_res_raw_NN)

    # overall_labels
    labels_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'], 
                            ['$m_x$', '$m_y$', '$m_{xy}$'], 
                            ['$v_{xz}$', '$v_{yz}$', '$v_{yz}$']])
    labels_eps = np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'], 
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_{xz}$', r'$\gamma_{xz}$', r'$\Delta$ $\gamma_{xz}$']])
    labels_u = np.array([[r'$u_x$',r'$u_y$',r'$u_z$'],
                            [r'$\vartheta_x$',r'$\vartheta_y$',r'$\vartheta_z$']])

    # defining values which vary among sig, eps and u
    if type_err == 'max':
        delta_label = np.array([['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ '],
                                ['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ '],
                                ['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ ']])
    elif type_err == 'rel':
        delta_label = np.array([['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ '],
                                ['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ '],
                                ['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ ']])
    elif type_err == 'rrse':
        delta_label = np.array([['$rRSE $ ', '$rRSE $ ', '$rRSE $ '],
                                ['$rRSE $ ', '$rRSE $ ', '$rRSE $ '],
                                ['$rRSE $ ', '$rRSE $ ', '$rRSE $ ']])
    elif type_err == 'nrse':
        delta_label = np.array([['$nRSE $ ', '$nRSE $ ', '$nRSE $ '],
                                ['$nRSE $ ', '$nRSE $ ', '$nRSE $ '],
                                ['$nRSE $ ', '$nRSE $ ', '$nRSE $ ']])

    if id == 'dsig':
        # fig = plt.figure()

        name = np.array([['Nx', 'Ny', 'Nxy'], 
                         ['Mx', 'My', 'Mxy'],
                         ['Qx', 'Qy', 'Qy']])
        
        labels = np.char.add(delta_label,labels_sig)
        num_rows = 3
        label_colorbar = np.array(['$[\%]$', '$[\%]$', '$[\%]$'])
        img = [0, 0, 0]

    if id == 'deps':
        name = np.array([['epsx', 'epsy', 'epsxy'], 
                        ['chix', 'chiy', 'chixy'],
                        [ 'gamx', 'gamy', 'gamy']])
        
        labels = np.char.add(delta_label,labels_eps)
        num_rows = 3
        label_colorbar = np.array(['$[\%]$', '$[\%]$', '$[\%]$'])
        img = [0, 0, 0]

        # Convert [-] strains to [permille] strains
        for i in range(3):
            for j in range(3):
                if i == 2 and j==2:
                    pass
                else:
                    colors[str(0)][name[i,j]][2] = colors[str(0)][name[i,j]][2]*10**(3)
                    colors_NN[str(0)][name[i,j]][2] = colors_NN[str(0)][name[i,j]][2]*10**(3)

    if id == 'du':
        name = np.array([['ux','uy','uz'],
                        ['thx','thy','thz']])
        labels = np.char.add(delta_label[0:2, :],labels_u)
        num_rows = 2
        label_colorbar = np.array(['$[\%]$', '$[\%]$'])
        img = [0, 0]


    # Convert mm to m (for the x-/ y- axes)
    for i in range(num_rows):
        for j in range(3):
            if i == 2 and j==2:
                pass
            else:
                colors[str(0)][name[i,j]][0] = colors[str(0)][name[i,j]][0]*10**(-3)
                colors[str(0)][name[i,j]][1] = colors[str(0)][name[i,j]][1]*10**(-3)
    
    # Calculate errors and determine ranges for colorbars
    min_temp, max_temp = np.zeros((name.shape)), np.zeros((name.shape))
    vmin, vmax = np.zeros((name.shape)), np.zeros((name.shape))
    sz = colors[str(0)][name[0,0]][2].shape
    diff, diff_perc_rel = np.zeros((sz[0], sz[1], name.shape[0], name.shape[1])), np.zeros((sz[0], sz[1], name.shape[0], name.shape[1]))
    rrse, nrse = np.zeros((sz[0], sz[1], name.shape[0], name.shape[1])), np.zeros((sz[0], sz[1], name.shape[0], name.shape[1]))
    max_, diff_perc_max = np.zeros_like((diff)), np.zeros_like((diff))
    aux_, aux__= np.zeros((sz[0], sz[1], name.shape[0], name.shape[1])), np.zeros((sz[0], sz[1], name.shape[0], name.shape[1]))
    for i in range(num_rows):
        for j in range(3):
            diff[:, :, i, j] = abs(colors[str(0)][name[i,j]][2]-colors_NN[str(0)][name[i,j]][2])
            colors[str(0)][name[i,j]][2][colors[str(0)][name[i,j]][2] == 0] = 1e-9
            if type_err == 'rel':
                diff_perc_rel[:, :, i, j] = np.divide(diff[:, :, i, j], abs(colors[str(0)][name[i,j]][2]))*100
            elif type_err == 'max':
                col_aux = abs(colors[str(0)][name[i,j]][2])
                col_aux_nn = abs(colors_NN[str(0)][name[i,j]][2])
                for k in range(sz[0]):
                    for l in range(sz[1]):
                        max_[k, l, i, j] = max(col_aux[k,l], col_aux_nn[k,l])
                diff_perc_max[:,:, i,j] = np.divide(diff[:, :, i, j], max_[:,:,i,j])*100
            elif type_err == 'rrse':
                diff[:,:,i,j] = np.sqrt((colors[str(0)][name[i,j]][2] - colors_NN[str(0)][name[i,j]][2])**2)
                if i+3*j < 8:
                    mean = stats['stats_y_train']['mean'][i+3*j]*np.ones_like(colors[str(0)][name[i,j]][2])
                else: 
                    mean == 0*np.ones_like(colors[str(0)][name[i,j]][2])
                aux_[:,:,i,j] = np.sqrt((colors[str(0)][name[i,j]][2] - mean) ** 2)
                aux_[:,:,i,j][aux_[:,:,i,j] == 0] = 1
                rrse[:,:,i,j] = np.divide(diff[:,:,i,j], aux_[:,:,i,j])*100
            elif type_err == 'nrse':
                diff[:,:,i,j] = np.sqrt((colors[str(0)][name[i,j]][2] - colors_NN[str(0)][name[i,j]][2])**2)
                if i+3*j < 8:
                    q_5 = stats['stats_y_train']['q_5'][i+3*j]
                    q_95 = stats['stats_y_train']['q_95'][i+3*j]
                    aux__[:,:,i,j] = (q_95-q_5)*np.ones_like(colors[str(0)][name[i,j]][2])
                else: 
                    aux__[:,:,i,j] = np.ones_like(colors[str(0)][name[i,j]][2])
                nrse[:,:,i,j] = np.divide(diff[:,:,i,j], aux__[:,:,i,j])*100

            # assign to plot_vector
            if type_err == 'rel':
                diff_perc = diff_perc_rel
            elif type_err == 'max':
                diff_perc = diff_perc_max
            elif type_err == 'rrse':
                diff_perc = rrse
            elif type_err == 'nrse':
                diff_perc = nrse

            # Find ranges for colorbar
            min_temp[i,j] = np.min(diff_perc[:, :, i, j])
            max_temp[i,j] = np.max(diff_perc[:, :, i, j])
        # vmin[i,:] = np.min(min_temp[i,:])*np.ones((1,3))
        # vmax[i,:] = np.max(max_temp[i,:])*np.ones((1,3))
        vmin[i,:] = 0*np.ones((1,3))
        vmax[i,:] = 10*np.ones((1,3))
    


    # plot figure
    fig, ax = plt.subplots(num_rows, 3, figsize = [8,num_rows*2.7])
    for i in range(num_rows):
        for j in range(3):
            if i == 2 and j==2:
                ax[i,j].set_title(' ')
            else:
                # if np.all((diff_perc[:,:,i,j]).astype(int) == 0):
                #         img[i] = ax[i,j].imshow(np.zeros_like((diff_perc[:,:,i,j])), 
                #                                 extent=(colors[str(0)][name[i, j]][0].min(), colors[str(0)][name[i, j]][0].max(), 
                #                                 colors[str(0)][name[i, j]][1].min(), colors[str(0)][name[i, j]][1].max()), 
                #                                 origin='lower', cmap="viridis", vmin=0, vmax=1e-9)
                #         ax[i,j].set_title(labels[i,j])
                # else:
                ax[i,j].contour(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], diff_perc[:, :, i, j],
                                        colors = 'black', linewidths = 0.5,
                                        vmin=vmin[i,j], vmax=vmax[i,j], 
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                img[i] = ax[i,j].contourf(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], diff_perc[:, :, i, j],
                                        vmin=vmin[i,j], vmax=vmax[i,j],
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                ax[i,j].set_title(labels[i,j])

    for i in range(num_rows):
        cbar = fig.colorbar(img[i], label = label_colorbar[i], ax = ax[i,0:3], location= 'right')
        for label in cbar.ax.get_yticklabels()[1::2]:
            label.set_visible(False)
    if num_rows == 3:
        ax[-1, -1].axis('off')
    for i in range(num_rows):
        for j in range(3):
            ax[i,j].set_aspect('equal', 'box')
            ax[i,j].axis('square')
    

    plt.savefig(os.path.join(path, 'single_'+ id +'.png'), format = 'png')
    plt.show()
    plt.close()


    return



def main_plt_comp(mat_res_raw, mat_res_raw_NN, id:str, path, type_err, stats, comp_list, paper=False):
    '''
    comparison plot with contour
    mat_res_raw     (dict)      original data (NLFEA)
    mat_res_raw_NN  (dict)      predicted data (NN)
    id              (str)       sig or eps
    path            (str)       save path
    type_err        (str)       type of error to show in the third row of contour plot (max, rel, rrse or nrse)
                                for u: only max or rel possible (no underlying dataset available)
    stats           (dict)      statistics of the dataset used for training
    comp_list       (list)      list of forces to compare
    paper           (bool)      create plot for paper-ready version

    
    '''
    if not paper: 
        # leave the units at Nmm/mm and N/mm
        colors = plot_colors(mat_res_raw)
        colors_NN = plot_colors(mat_res_raw_NN)
    elif paper: 
        # Change units of moment in mat_res_raw, to get kNm/m
        mat_res_raw_pap = mat_res_raw.copy() 
        mat_res_raw_NN_pap = mat_res_raw_NN.copy()
        factor = 10**(-3)
        for i in ['Mx', 'My', 'Mxy']:
            mat_res_raw_pap['POST'][i] = [factor*x for x in mat_res_raw['POST'][i]]
            mat_res_raw_NN_pap['POST'][i] = [factor*x for x in mat_res_raw_NN['POST'][i]]
        colors = plot_colors(mat_res_raw_pap)
        colors_NN = plot_colors(mat_res_raw_NN_pap)

    labels_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'], 
                            ['$m_x$', '$m_y$', '$m_{xy}$'], 
                            ['$v_{xz}$', '$v_{yz}$', '$v_{yz}$']])
    
    labels_eps = np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'],
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_{xz}$', r'$\gamma_{xz}$', r'$\Delta$ $\gamma_{xz}$']])
    
    labels_u = np.array([[r'$u_x$', r'$u_y$', r'$u_z$'],
                         [r'$\theta_x$', r'$\theta_y$', r'$\theta_z$']])
    
    label_add_nn = np.array([['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$']])
    
    label_add_fea = np.array([['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$']])
    
    if type_err == 'max':
        delta_label = np.array([['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ '],
                                ['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ '],
                                ['$\Delta_{max} $ ', '$\Delta_{max} $ ', '$\Delta_{max} $ ']])
    elif type_err == 'rel':
        delta_label = np.array([['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ '],
                                ['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ '],
                                ['$\Delta_{rel} $ ', '$\Delta_{rel} $ ', '$\Delta_{rel} $ ']])
    elif type_err == 'rrse':
        delta_label = np.array([['$rRSE $ ', '$rRSE $ ', '$rRSE $ '],
                                ['$rRSE $ ', '$rRSE $ ', '$rRSE $ '],
                                ['$rRSE $ ', '$rRSE $ ', '$rRSE $ ']])
    elif type_err == 'nrse':
        delta_label = np.array([['$nRSE $ ', '$nRSE $ ', '$nRSE $ '],
                                ['$nRSE $ ', '$nRSE $ ', '$nRSE $ '],
                                ['$nRSE $ ', '$nRSE $ ', '$nRSE $ ']])

    name = np.array(comp_list)
    num_cols = len(comp_list)


    if id == 'sig':
        name_sig = np.array([['Nx', 'Ny', 'Nxy'], 
                            ['Mx', 'My', 'Mxy'],
                            ['Qx', 'Qy', 'Qy']])
        name_sig_ext = np.array(['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy', 'Qx', 'Qy'])
        match_loc = np.isin(name_sig, name)
        match_loc[2,2] = False
        labels_comp = np.zeros((3, num_cols), dtype=object)
        indices = np.where(np.isin(name_sig_ext, name))[0]

        sgl = np.array([['$[N/mm]$'],
                    ['$[N/mm]$'],
                    ['$[\%]$']])
        
        for i in range(num_cols):
            labels_comp[0,i] = np.char.add(labels_sig[match_loc][i], label_add_fea[match_loc][i])
            labels_comp[1,i] = np.char.add(labels_sig[match_loc][i], label_add_nn[match_loc][i])
            labels_comp[2,i] = np.char.add(delta_label[match_loc][i], labels_sig[match_loc][i])
    
        # label_colorbar = sgl
        # label_colorbar = np.hstack([sgl]*num_cols)
        if num_cols == 3:
            label_colorbar = np.array([['$[N/mm]$', '$[Nmm/mm]$', '$[N/mm]$'],
                                    ['$[N/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$'],
                                    ['$[\%]$', '$[\%]$', '$[\%]$']
                                    ])
        elif num_cols == 8:
            label_colorbar = np.array([ ['$[N/mm]$', '$[N/mm]$','$[N/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$', '$[N/mm]$', '$[N/mm]$'], 
                                        ['$[N/mm]$', '$[N/mm]$','$[N/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$', '$[N/mm]$', '$[N/mm]$'],
                                        [r'$[%]$', r'$[%]$', r'$[%]$', r'$[%]$', r'$[%]$', r'$[%]$', r'$[%]$', r'$[%]$']
                                    ])
        elif num_cols == 1: 
            label_colorbar = np.array([['$[N/mm]$', '$[Nmm/mm]$', '$[N/mm]$'],
                                    ['$[N/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$'],
                                    ['$[\%]$', '$[\%]$', '$[\%]$']
                                    ])
    
    elif id == 'eps':
        name_eps = np.array([['epsx', 'epsy', 'epsxy'],
                             ['chix', 'chiy', 'chixy'],
                             ['gamx', 'gamy', 'gamy']])
        name_eps_ext = np.array(['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy'])
        match_loc = np.isin(name_eps, name)
        match_loc[2,2] = False
        labels_comp = np.zeros((3, num_cols), dtype=object)
        indices = np.where(np.isin(name_eps_ext, name))[0]
        
        for i in range(num_cols):
            labels_comp[0,i] = np.char.add(labels_eps[match_loc][i], label_add_fea[match_loc][i])
            labels_comp[1,i] = np.char.add(labels_eps[match_loc][i], label_add_nn[match_loc][i])
            labels_comp[2,i] = np.char.add(delta_label[match_loc][i], labels_eps[match_loc][i])

        label_colorbar = np.array([[r'$[mm/mm]$', r'$[1/mm]$', r'$[mm/mm]$'],
                                [r'$[mm/mm]$', r'$[1/mm]$', r'$[mm/mm]$'],
                                ['$[\%]$', '$[\%]$', '$[\%]$']])
        
    elif id == 'u':
        # nRSE and rRSE values cannot be calculated use 
        if type_err == 'nrse' or type_err == 'rrse':
            raise UserWarning('These errors cannot be calculated for the displacement plots. Please choose either "max" or "rel" as error_type.')       

        name_u = np.array([['ux', 'uy', 'uz'],
                             ['thx', 'thy', 'thz']])
        name_u_ext = np.array(['ux', 'uy', 'uz', 'thx', 'thy', 'thz'])
        match_loc = np.isin(name_u, name)
        labels_comp = np.zeros((3, num_cols), dtype=object)
        indices = np.where(np.isin(name_u_ext, name))[0]
        
        for i in range(num_cols):
            labels_comp[0,i] = np.char.add(labels_u[match_loc][i], label_add_fea[:2,:][match_loc][i])
            labels_comp[1,i] = np.char.add(labels_u[match_loc][i], label_add_nn[:2,:][match_loc][i])
            labels_comp[2,i] = np.char.add(delta_label[:2,:][match_loc][i], labels_u[match_loc][i])

        label_colorbar = np.array([[r'$[mm]$', r'$[mm]$', r'$[mm]$'],
                                    [r'$[mm]$', r'$[mm]$', r'$[mm]$'],
                                    ['$[\%]$', '$[\%]$', '$[\%]$']])


    # Convert mm to m (for the x-/ y- axes)
    for i in range(num_cols):
        colors[str(0)][name[i]][0] = colors[str(0)][name[i]][0]*10**(-3)
        colors[str(0)][name[i]][1] = colors[str(0)][name[i]][1]*10**(-3)


    # Calculate errors and determine ranges for colorbars
    # min_temp, max_temp = np.zeros((name.shape)), np.zeros((name.shape))
    # vmin, vmax = np.zeros((name.shape)), np.zeros((name.shape))
    sz = colors[str(0)][name[0]][2].shape
    diff, diff_perc_rel = np.zeros((sz[0], sz[1], name.shape[0])), np.zeros((sz[0], sz[1], name.shape[0]))
    max_, diff_perc_max = np.zeros_like((diff)), np.zeros_like((diff))
    aux_, aux__, nrse, rrse = np.zeros_like((diff)), np.zeros_like((diff)), np.zeros_like((diff)), np.zeros_like((diff))
    for i in range(num_cols):
        diff[:, :, i] = abs(colors[str(0)][name[i]][2]-colors_NN[str(0)][name[i]][2])
        colors[str(0)][name[i]][2][colors[str(0)][name[i]][2] == 0] = 1e-9
        if type_err == 'rel':
            diff_perc_rel[:, :, i] = np.divide(diff[:, :, i], abs(colors[str(0)][name[i]][2]))*100
        elif type_err == 'max':
            col_aux = abs(colors[str(0)][name[i]][2])
            col_aux_nn = abs(colors_NN[str(0)][name[i]][2])
            for k in range(sz[0]):
                for l in range(sz[1]):
                    max_[k, l, i] = max(col_aux[k,l], col_aux_nn[k,l])
            diff_perc_max[:,:, i] = np.divide(diff[:, :, i], max_[:,:,i])*100
        elif type_err == 'rrse':
            diff[:,:,i] = np.sqrt((colors[str(0)][name[i]][2] - colors_NN[str(0)][name[i]][2])**2)
            if paper: 
                mean = np.ones_like(colors[str(0)][name[i]][2])
                if (i>2) and (i<6):
                    mean[i] = stats['stats_y_train']['mean'][indices[i]]*10**(-3)
                else: 
                    mean[i] = stats['stats_y_train']['mean'][indices[i]]
            elif not paper: 
                mean = stats['stats_y_train']['mean'][indices[i]]*np.ones_like(colors[str(0)][name[i]][2])
            aux_[:,:,i] = np.sqrt((colors[str(0)][name[i]][2] - mean) ** 2)
            aux_[:,:,i][aux_[:,:,i] == 0] = 1
            rrse[:,:,i] = np.divide(diff[:,:,i], aux_[:,:,i])*100
        elif type_err == 'nrse':
            if stats['stats_y_train']['q_5'].shape[0] < 72 and len(comp_list)>1:
                raise UserWarning('When using a ONEDIM NN, please do not plot other variables with nRSE error, just use e.g. comp_list = [\'epsx\'] or comp_list = [\'Nx\']')
            diff[:,:,i] = np.sqrt((colors[str(0)][name[i]][2] - colors_NN[str(0)][name[i]][2])**2)
            if paper and (i>2) and (i<6): 
                q_5 = stats['stats_y_train']['q_5'][indices[i]]*10**(-3)
                q_95 = stats['stats_y_train']['q_95'][indices[i]]*10**(-3)
            elif paper: 
                q_5 = stats['stats_y_train']['q_5'][indices[i]]
                q_95 = stats['stats_y_train']['q_95'][indices[i]]
            else: 
                if i < 3 or i > 6:
                    if id == 'eps':
                        q_5 = stats['stats_X_train']['q_5'][indices[i]]
                        q_95 = stats['stats_X_train']['q_95'][indices[i]]
                    elif id == 'sig':
                        q_5 = stats['stats_y_train']['q_5'][indices[i]]*10**(5)
                        q_95 = stats['stats_y_train']['q_95'][indices[i]]*10**(5)
                else: 
                    if id == 'eps':
                        q_5 = stats['stats_X_train']['q_5'][indices[i]]*10**(-1)
                        q_95 = stats['stats_X_train']['q_95'][indices[i]]*10**(-1)
                    else: 
                        q_5 = stats['stats_y_train']['q_5'][indices[i]]*10**(6)
                        q_95 = stats['stats_y_train']['q_95'][indices[i]]*10**(6)
            aux__[:,:,i] = (q_95-q_5)*np.ones_like(colors[str(0)][name[i]][2])
            nrse[:,:,i] = np.divide(diff[:,:,i], aux__[:,:,i])*100

        # assign to plot_vector
        if type_err == 'rel':
            diff_perc = diff_perc_rel
        elif type_err == 'max':
            diff_perc = diff_perc_max
        elif type_err == 'rrse':
            diff_perc = rrse
        elif type_err == 'nrse':
            diff_perc = nrse

    # Find ranges of colorbar
    vmin_row1 = []
    vmax_row1 = []
    vmin_row2 = []
    vmax_row2 = []
    vmin_r = []
    vmax_r = []
    for j in range(num_cols):
        z_row1 = np.array([colors[str(0)][name[j]][2], colors_NN[str(0)][name[j]][2], diff_perc[:,:,j]])[0]
        vmin_row1.append(np.min(z_row1))
        vmax_row1.append(np.max(z_row1))
        z_row2 = np.array([colors[str(0)][name[j]][2], colors_NN[str(0)][name[j]][2], diff_perc[:,:,j]])[1]
        vmin_row2.append(np.min(z_row2))
        vmax_row2.append(np.max(z_row2))
        vmin_r.append(min(vmin_row1[j], vmin_row2[j]))
        vmax_r.append(max(vmax_row1[j], vmax_row2[j]))

    if type_err == 'nrse':
        vmax_perc = 10
    elif type_err == 'max' or type_err == 'rel':
        vmax_perc = 50
    else: 
        vmax_perc = 10
    
    
    # Plot figure
    if not paper:
        fig, ax = plt.subplots(3, num_cols, figsize = [num_cols*4, 8])
        img = np.empty((3, num_cols), dtype=object)
        ax, img = np.atleast_2d(ax).reshape(3,num_cols), np.atleast_2d(img).reshape(3, num_cols)

        for i in range(3):
            for j in range(num_cols):
                z = np.array([colors[str(0)][name[j]][2],
                    colors_NN[str(0)][name[j]][2],
                    diff_perc[:,:,j]
                    ])
                
                if i <2:
                    img[i,j] = ax[i,j].contourf(colors[str(0)][name[j]][0], colors[str(0)][name[j]][1], z[i],
                                                vmin=vmin_r[j], vmax=vmax_r[j],
                                                levels = get_good_labels(vmin_r[j], vmax_r[j])
                                                )
                    ax[i,j].contour(colors[str(0)][name[j]][0], colors[str(0)][name[j]][1], z[i],
                                        colors = 'black', linewidths = 0.5,
                                        vmin=vmin_r[j], vmax=vmax_r[j],
                                        levels = get_good_labels(vmin_r[j], vmax_r[j])
                                        )
                elif i == 2: 
                    img[i,j] = ax[i,j].contourf(colors[str(0)][name[j]][0], colors[str(0)][name[j]][1], z[i], 
                                                vmin=0, vmax=vmax_perc,
                                                levels = np.linspace(0, vmax_perc, vmax_perc+1))
                    ax[i,j].contour(colors[str(0)][name[j]][0], colors[str(0)][name[j]][1], z[i],
                                        colors = 'black', linewidths = 0.5,
                                        vmin=0, vmax=vmax_perc,
                                        levels = np.linspace(0, vmax_perc, vmax_perc+1)
                                        )
                ax[i,j].set_title(labels_comp[i,j])

        for i in range(3):
            for j in range(num_cols):
                if label_colorbar is not None:
                    cbar = fig.colorbar(img[i,j], label = label_colorbar[i,j], ax = ax[i,j]) # , location= 'right')
        for i in range(3):
            for j in range(num_cols):
                ax[i,j].set_aspect(aspect = 'equal', adjustable = 'box')
                ax[i,j].axis('square')

        plt.subplots_adjust(hspace = 0.4)
        plt.savefig(os.path.join(path, 'comparison_'+ str(comp_list) +'.png'), format = 'png')
        # plt.show()
        # plt.close()


    elif paper:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman",
            "font.size": 9,
            "text.latex.preamble": r'\usepackage{wasysym}'
            })

        plt.rc('text.latex', preamble=
            r'\usepackage{amsmath}' + 
            r'\usepackage{times}')
    
        label_colorbar = np.array([['[kN/m]', '[\%]'],
                                    ['[kNm/m]', '[\%]'], 
                                    ['[kN/m]', '[\%]']])
    
        # adjust the vmin, vmax values again, such that they are the same for all n, the same for all m, the same for all v
        vmax_r_p = []
        vmin_r_p = []
        # Normal force
        vmin_r_p_n = min(vmin_r[0:3])
        vmax_r_p_n = max(vmax_r[0:3])
        vmin_r_p.append(vmin_r_p_n)
        vmax_r_p.append(vmax_r_p_n)
        # Moment
        vmin_r_p_m = min(vmin_r[3:6])
        vmax_r_p_m = max(vmax_r[3:6])
        vmin_r_p.append(vmin_r_p_m)
        vmax_r_p.append(vmax_r_p_m)
        # Shear force
        vmin_r_p_v = min(vmin_r[6:8])
        vmax_r_p_v = max(vmax_r[6:8])
        vmin_r_p.append(vmin_r_p_v)
        vmax_r_p.append(vmax_r_p_v)

        # subplot definition

        def create_subplots_contour(ax, row_o, col_o, num_rows_p, num_cols_p, vmin_r_p, vmax_r_p):
            img = [[0] * num_cols_p] * num_rows_p
            for i in range(num_rows_p):
                for j in range(num_cols_p):
                    z = np.array([colors[str(0)][name[i+row_o*3]][2],
                        colors_NN[str(0)][name[i+row_o*3]][2],
                        diff_perc[:,:,i+row_o*3]
                        ])
                    
                    img[i][j] = ax[i][j].contourf(colors[str(0)][name[i+row_o*3]][0], colors[str(0)][name[i+row_o*3]][1], z[j],
                                                # vmin=vmin_r[i+row_o*3], vmax=vmax_r[i+row_o*3],
                                                # levels = get_good_labels(vmin_r[i+row_o*3], vmax_r[i+row_o*3])
                                                vmin = vmin_r_p[row_o], vmax = vmax_r_p[row_o],
                                                levels = get_good_labels(vmin_r_p[row_o], vmax_r_p[row_o])
                                                )
                    ax[i][j].contour(colors[str(0)][name[i+row_o*3]][0], colors[str(0)][name[i+row_o*3]][1], z[j],
                                        colors = 'black', linewidths = 0.5,
                                        # vmin=vmin_r[i+row_o*3], vmax=vmax_r[i+row_o*3],
                                        # levels = get_good_labels(vmin_r[i+row_o*3], vmax_r[i+row_o*3])
                                        vmin = vmin_r_p[row_o], vmax = vmax_r_p[row_o],
                                        levels = get_good_labels(vmin_r_p[row_o], vmax_r_p[row_o])
                                        )
                        
                    ax[i][j].set_aspect(aspect = 'equal', adjustable = 'box')
                    ax[i][j].axis('square')

                    # Tick management:
                    ax[i][j].set_xticks([min(colors[str(0)][name[i+row_o*3]][0][0]), max(colors[str(0)][name[i+row_o*3]][0][0])])
                    ax[i][j].set_yticks([min(colors[str(0)][name[i+row_o*3]][0][0]), max(colors[str(0)][name[i+row_o*3]][0][0])])
                    dy = 3/72
                    offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                    label = ax[i][j].yaxis.get_majorticklabels()
                    label[0].set_transform(label[0].get_transform()+offset)
                    label[1].set_transform(label[1].get_transform()-offset)

                    # ax[i][j].draw(renderer = fig.canvas.get_renderer())
                    

                    if i < num_rows_p-1:  
                        # Hide x-axis labels for top and middle rows
                        ax[i][j].set_xticklabels([])
                        ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    if j > 0:  
                    # Hide y-axis labels for all but the leftmost column
                        ax[i][j].set_yticklabels([])
                        ax[i][j].tick_params(axis="y", which="both", left=False, right=False)

                    

            return img
   

        def create_subplots_error(ax, row_o, col_o, num_rows_p, num_cols_p):
            img = np.empty((num_rows_p, num_cols_p), dtype=object)
            for i in range(num_rows_p):
                for j in range(num_cols_p):
                    z = np.array([colors[str(0)][name[i+row_o*3]][2],
                        colors_NN[str(0)][name[i+row_o*3]][2],
                        diff_perc[:,:,i+row_o*3]
                        ])

                    img[i,j] = ax[i][j].contourf(colors[str(0)][name[i+row_o*3]][0], colors[str(0)][name[i+row_o*3]][1], z[2], 
                                                            vmin=0, vmax=vmax_perc,
                                                            levels = np.linspace(0, vmax_perc, 21))
                    ax[i][j].contour(colors[str(0)][name[i+row_o*3]][0], colors[str(0)][name[i+row_o*3]][1], z[2],
                                colors = 'black', linewidths = 0.5,
                                vmin=0, vmax=vmax_perc,
                                levels = np.linspace(0, vmax_perc, 21)
                                )
                    ax[i][j].set_aspect(aspect = 'equal', adjustable = 'box')
                    ax[i][j].axis('square')

                    # Tick management:
                    ax[i][j].set_xticks([min(colors[str(0)][name[i+row_o*3]][0][0]), max(colors[str(0)][name[i+row_o*3]][0][0])])
                    ax[i][j].set_yticks([min(colors[str(0)][name[i+row_o*3]][0][0]), max(colors[str(0)][name[i+row_o*3]][0][0])])
                    dy = 3/72
                    offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
                    label = ax[i][j].yaxis.get_majorticklabels()
                    label[0].set_transform(label[0].get_transform()+offset)
                    label[1].set_transform(label[1].get_transform()-offset)

                    if i < num_rows_p-1:  
                    # Hide x-axis labels for top and middle rows
                        ax[i][j].set_xticklabels([])
                        ax[i][j].tick_params(axis="x", which="both", bottom=False, top=False)
                    if j > 0:  
                    # Hide y-axis labels for all but the leftmost column
                        ax[i][j].set_yticklabels([])
                        ax[i][j].tick_params(axis="y", which="both", left=False, right=False)

            return img

        # fig, ax = plt.subplots(num_rows_p, num_cols_p, figsize = (14/2.54, 18/2.54))
        fig = plt.figure(figsize = (10/2.54, 18/2.54))
        outer_grid = gridspec.GridSpec(3, 2, wspace=0.5, hspace=0.2, 
                                       width_ratios=[2, 1],
                                       height_ratios=[3, 3, 2])
        
        img_all = np.empty((3, 2), dtype=object)
        for i in range(3):
            for j in range(2):
                if i<2 and j<1:                
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        3,2, subplot_spec=outer_grid[i,j], wspace=0, hspace=0.1
                    )
                elif i<2 and j==1:
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        3,1, subplot_spec=outer_grid[i,j], wspace=0, hspace=0.1
                    )
                elif i==2 and j<1:
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        2,2, subplot_spec=outer_grid[i,j], wspace=0, hspace=0.1
                    )
                elif i==2 and j==1:
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        2,1, subplot_spec=outer_grid[i,j], wspace=0, hspace=0.1
                    )

                # Create space for plots
                nrows, ncols = inner_grid.get_geometry()
                axes = []
                for row in range(nrows):
                    axes_row = []
                    for col in range(ncols):
                        ax = fig.add_subplot(inner_grid[row, col])
                        # ax.set_aspect('equal', 'box')
                        # ax.axis('square')
                        for spine in ax.spines.values():
                            spine.set_linewidth(0.5)  # Set the width of the outline
                            spine.set_color('black')   # Set the color of the outline
                        ax.tick_params(axis='both', labelsize=7, length=2, width=0.25, color = 'lightgrey', labelcolor = 'grey')
                        xticklabels = ax.get_xticklabels()
                        xticklabels[0].set_x(xticklabels[0].get_position()[0]+0.2)
                        xticklabels[1].set_x(xticklabels[1].get_position()[0]-0.2)
                        axes_row.append(ax)                        
                    axes.append(axes_row)

                # Fill plots with desired data
                if j < 1:
                    if i<2:
                        img_all[i,j] = create_subplots_contour(axes, i, j, num_rows_p = 3, num_cols_p = 2, vmin_r_p=vmin_r_p, vmax_r_p=vmax_r_p)
                    elif i== 2:
                        img_all[i,j] = create_subplots_contour(axes, i, j, num_rows_p = 2, num_cols_p = 2, vmin_r_p=vmin_r_p, vmax_r_p=vmax_r_p)
                elif j == 1:
                    if i <2:
                        img_all[i,j] = create_subplots_error(axes, i, j, num_rows_p=3, num_cols_p=1)
                    elif i == 2:
                        img_all[i,j] = create_subplots_error(axes, i, j, num_rows_p=2, num_cols_p=1)
                                
                if i<2:
                    axins = inset_axes(axes[nrows-1][ncols-1], # here using axis of the lowest plot
                                        width="5%",  # width = 5% of parent_bbox width
                                        height="250%",  # height : 90% good for a (4x4) Grid
                                        loc='lower left',
                                        bbox_to_anchor=(1.05, 0.3, 1, 1),
                                        bbox_transform=axes[nrows-1][ncols-1].transAxes,
                                        borderpad=0,
                                        )
                else: 
                    axins = inset_axes(axes[nrows-1][ncols-1], # here using axis of the lowest plot
                                        width="5%",  # width = 5% of parent_bbox width
                                        height="150%",  # height : 90% good for a (4x4) Grid
                                        loc='lower left',
                                        bbox_to_anchor=(1.05, 0.3, 1, 1),
                                        bbox_transform=axes[nrows-1][ncols-1].transAxes,
                                        borderpad=0,
                                        )

                cbar = fig.colorbar(img_all[i,j][0][0], cax=axins)
                cbar.set_label(label = label_colorbar[i][j], fontsize=9)
                cbar.ax.tick_params(labelsize=7, width = 0.25)
                # cbar.outline.set_edgecolor('lightgrey')
                cbar.outline.set_linewidth(0.5)

        fig.text(0.22, 0.9, "(a) FEA", ha='center', va='center', fontsize=9)
        fig.text(0.43, 0.9, "(b) ML-FEA", ha='center', va='center', fontsize=9)
        fig.text(0.8, 0.9, r"(c) \textit{nRSE}", ha='center', va='center', fontsize=9)

        fig.text(0.05, 0.85, r'\textit{n}$_x$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.75, r'$\textit{n}_y$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.65, r'$\textit{n}_{xy}$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.55, r'$\textit{m}_x$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.45, r'$\textit{m}_y$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.35, r'$\textit{m}_{xy}$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.25, r'$\textit{v}_{xz}$', ha='center', va='center', fontsize=9)
        fig.text(0.05, 0.15, r'$\textit{v}_{yz}$', ha='center', va='center', fontsize=9)

        # plt.tight_layout()


        # plt.savefig(os.path.join(path, 'comparison_'+ 'all'+ '_paper.pdf'), format = 'pdf')
        plt.savefig(os.path.join(path, 'comparison_'+ 'all'+ '_paper.tif'), dpi=600)
        # plt.show()
        # plt.close()


    return





def get_good_labels(vmin, vmax, no_steps = 10):

    if vmin > vmax * 0.9:
        vmin = vmax * 0.9

    step = (vmax-vmin)/no_steps
    n_round = np.floor(np.log10(step))
    step_ = np.round(step/(10**n_round))*(10**n_round)
    
    vmin_f = np.floor(vmin/step_)*step_
    vmax_f = np.ceil(vmax/step_)*step_
    if abs(n_round) > 1e10:
        step_ = 1
        vmin_f = 0
        vmax_f = 1
    num_steps = int((vmax_f-vmin_f)/step_)+1

    # while num_steps > 2: 
    custom_levels = np.linspace(vmin_f, vmax_f, num_steps)
        # custom_levels = np.round(custom_levels, 6).astype(float)

        # if np.all(np.diff(custom_levels) > 0):
            # break
        # else: 
            # num_steps -= 1

    return custom_levels



def contour_perit(mat_res_raw, mat_res_NN, numit, mat_res_perm = None, idx = 2, save_path = None, diff = False, same = True, tag = 'u'):
    '''
    creates a contour plot for u per iteration in the NLFEA vs NN-NLFEA hybrid solver
    mat_res         (dict)          includes important output parameters NLFEA
    mat_res_NN      (dict)          includes important output parameters NN
    mat_res_perm    (dict)          includes important output parameters NLFEA with permutation
    numit           (int)           total amount of iterations
    idx             (int)           index of u to be plotted - can be 0...5 for ux, .. thz
                                    if it contains a list of two values (i,j): then plot stiffness entry at that index instead.    
    save_path       (str)           path to save_location of plot
    diff            (bool)          if True: includes plot for differences between each iteration step      
    same            (bool)          if True: use same colorbar scaling (min, max) for NN and NLFEA     
    tag             (str)           identifier for which contour to be plotted (sig, eps, u or D)
    '''

    # mat_res_raw0 = pd.DataFrame.from_dict(mat_res_raw)
    # mat_res_raw0_ = mat_res_raw0.loc[0,:]
    mat_res_raw0_ = mat_res_raw
    if mat_res_NN is not None:
        # mat_res_NN0 = pd.DataFrame.from_dict(mat_res_NN)
        # mat_res_NN0_ = mat_res_NN0.loc[0,:]
        mat_res_NN0_ = mat_res_NN
    else: 
        mat_res_NN0_ = None

    if mat_res_perm is not None: 
       mat_res_perm0 = pd.DataFrame.from_dict(mat_res_raw)
       mat_res_perm0_ = mat_res_perm0.loc[0,:]
       mat_res = {
            '0':mat_res_raw0_,
            '1':mat_res_NN0_,
            '2': mat_res_perm0_
        }
    else:
        mat_res = {
            '0':mat_res_raw0_,
            '1':mat_res_NN0_,
        }

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw0_)

    # define x, y- values for the grid
    colors={}
    xa = np.array(COORD['n'][2][0][:, 0]).reshape(len(ux), 1)[MASK[0]].flatten()
    ya = np.array(COORD['n'][2][0][:, 1]).reshape(len(ux), 1)[MASK[0]].flatten()

    # define z-values for the contour (in terms of colour)
    if tag == 'u':
        # create u-plots
        u_all = np.zeros((len(mat_res),6,numit,int(mat_res['0']['u_cum'].shape[1]/6)))
        for i in range(len(mat_res)):
            for k in range(6):
                u_i = mat_res[str(i)]['u_cum'][:,k::6]
                u_all[i, k, :,:] = u_i[:numit,:].reshape((numit,-1))
    else: 
        # create D-plots - no adjustment needed
        pass

    # create colors for u or D
    nrows = int(len(mat_res)*2) if diff else int(len(mat_res))
    ncols = numit
    for i in range(nrows): 
        for j in range(numit):
            l = int(i/2) if diff else i 
            # --- colors for u-plots ------
            if tag == 'u':
                v = u_all[l][idx[0]]     
                # for plotting differences too
                if diff and i%2 != 0:
                    if j == 0:
                        colors[str(i)+str(j)] = colors[str(i-1)+str(j)]
                    else:
                        colors[str(i)+str(j)] = [0,0,0]
                        colors[str(i)+str(j)][0] = colors[str(i-1)+str(j)][0]
                        colors[str(i)+str(j)][1] = colors[str(i-1)+str(j)][1]
                        colors[str(i)+str(j)][2] = np.abs(np.divide((colors[str(i-1)+str(j-1)][2]-colors[str(i-1)+str(j)][2]),
                                                                    colors[str(i-1)+str(j)][2]))
                # for only plotting contour u plots
                else:
                    colors[str(i)+str(j)] = colorgrids(xa,ya,v[j][MASK[0]].flatten(),0, mat_res[str(l)])
            
            # ---- colors for D-plots -------
            elif tag == 'D': 
                if mat_res[str(l)] is not None:
                    v = mat_res[str(l)]['De_cum'][:,:,0,0,idx[0],idx[1]]    # for D plots
                    colors[str(i)+str(j)] = colorgrids(COORD['c'][3][0][:,0],COORD['c'][3][0][:,1], v[j], 0, mat_res[str(l)])
                else: 
                    colors[str(i)+str(j)] = None
            # ---- colors for sig-plots -------
            elif tag == 'sig': 
                v = mat_res[str(l)]['sh_cum'][:,:,0,0,idx[0]]    # for sig plots
                colors[str(i)+str(j)] = colorgrids(COORD['c'][3][0][:,0],COORD['c'][3][0][:,1], v[j], 0, mat_res[str(l)])
            # ---- colors for eps-plots -------
            elif tag == 'eps':
                v = mat_res[str(l)]['eh_cum'][:,:,0,0,idx[0]]    # for eps plots
                colors[str(i)+str(j)] = colorgrids(COORD['c'][3][0][:,0],COORD['c'][3][0][:,1], v[j], 0, mat_res[str(l)])

    # preparations for the figure
    if diff:
        labels_y = ['NLFEA', r'$\partial u_{NLFEA}/u_{NLFEA}$', 'NN', r'$\partial u_{NN}/u_{NN}$', 'PERM',  r'$\partial u_{PERM}/u_{PERM}$']
    else:
        labels_y = ['NLFEA', 'NN', 'PERM']
    if tag == 'u':
        labels_u = ['ux', 'uy', 'uz', 'thx', 'thy', 'thz']
        label_colorbar_u = ['mm', 'mm', 'mm', '1/mm', '1/mm', '1/mm']
        labels = [labels_u[idx[0]] + str(_ + 1) for _ in range(numit)]
        labels_diff = ['d'+ labels_u[idx[0]] + str(_ + 1) for _ in range(numit)]
        label_colorbar = [label_colorbar_u[idx[0]] + str(_ + 1) for _ in range(nrows)] if not diff else ['mm', '%', 'mm', '%', 'mm', '%']
    elif tag == 'D': 
        label_colorbar = ['N,mm']*nrows
        labels = ['D_' +str(idx[0]) + str(idx[1]) +',' + str(_ + 1) for _ in range(numit)]
        labels_diff = labels
    elif tag == 'sig': 
        label_colorbar = ['N,mm']*nrows
        labels = ['sig_' +str(idx[0]) +',' + str(_ + 1) for _ in range(numit)]
        labels_diff = labels
    elif tag == 'eps': 
        label_colorbar = ['[-, 1/mm]']*nrows
        labels = ['eps_' +str(idx[0]) +',' + str(_ + 1) for _ in range(numit)]
        labels_diff = labels
    
    # Convert mm to m (for the x-/ y- axes)
    for i in range(nrows):
        for j in range(ncols):
            if colors[str(i)+str(j)] is not None:
                colors[str(i)+str(j)][0] = colors[str(i)+str(j)][0]*10**(-3)
                colors[str(i)+str(j)][1] = colors[str(i)+str(j)][1]*10**(-3)


    # Define max, min for colorbar
    min_temp, max_temp = np.zeros((nrows, ncols)), np.zeros((nrows, ncols))
    vmin, vmax = np.zeros((nrows, ncols)), np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if colors[str(i)+str(j)] is not None:
                min_temp[i,j] = np.min(np.array(colors[str(i)+str(j)][2]))
                max_temp[i,j] = np.max(np.array(colors[str(i)+str(j)][2]))
        if i%2 != 0 and diff:
            vmin[i,:] = 0
            vmax[i,:] = 1
        else:
            vmin[i,:] = np.min(min_temp[i,:])*np.ones((1,ncols))
            vmax[i,:] = np.max(max_temp[i,:])*np.ones((1,ncols))
    if same:
        l = len(mat_res) if diff else int(len(mat_res)/2)
        vmin[0,:]   = np.minimum(vmin[0,:], vmin[l,:])
        vmin[l,:]   = np.minimum(vmin[0,:], vmin[l,:])
        if mat_res_perm is not None:
            vmin[0,:] = np.minimum(vmin[0,:], vmin[2*l,:])
            vmin[l,:] = np.minimum(vmin[l,:], vmin[2*l,:])
            vmin[2*l,:] = np.minimum(vmin[0,:], vmin[l,:], vmin[2*l,:])

        vmax[0,:]   = np.maximum(vmin[0,:], vmax[l,:])
        vmax[l,:]   = np.maximum(vmin[0,:], vmax[l,:])
        if mat_res_perm is not None:
            vmax[0,:] = np.maximum(vmax[0,:], vmax[2*l,:])
            vmax[l,:] = np.maximum(vmax[l,:], vmax[2*l,:])
            vmax[2*l,:] = np.maximum(vmin[0,:], vmax[l,:], vmax[2*l,:])

    img = [0]*nrows


    # plot the figure
    fig, axs = plt.subplots(nrows,ncols, figsize = [ncols*3, nrows*3])
    for i in range(nrows):
        for j in range(ncols):
            if diff and i%2 != 0 and j == 0:
                axs[i,j].set_title(' ')
                axs[i,j].axis('off')
            else:
                if colors[str(i)+str(j)] is not None:
                    axs[i,j].contour(colors[str(i)+str(j)][0], colors[str(i)+str(j)][1], colors[str(i)+str(j)][2],
                                            colors = 'black', linewidths = 0.5,
                                            vmin=vmin[i,j], vmax=vmax[i,j], 
                                            levels = get_good_labels(vmin[i,j], vmax[i,j])
                                            )
                    img[i] = axs[i,j].contourf(colors[str(i)+str(j)][0], colors[str(i)+str(j)][1], colors[str(i)+str(j)][2],
                                            vmin=vmin[i,j], vmax=vmax[i,j],
                                            levels = get_good_labels(vmin[i,j], vmax[i,j])
                                            )
                    axs[i,j].set_title(labels[j]) if i%2==0 else axs[i,j].set_title(labels_diff[j])
                    if j == 0: 
                        axs[i,j].set_ylabel(labels_y[i])
                else: 
                    img[i] = None
    for i in range(nrows):
        if img[i] is not None:
            cbar = fig.colorbar(img[i], label = label_colorbar[i], ax = axs[i, 0:ncols], location= 'right')
            for label in cbar.ax.get_yticklabels()[::2]:
                label.set_visible(False)
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].set_aspect('equal', 'box')
            axs[i,j].axis('square')
   
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'contour_perit_'+tag+str(idx)+'.png'), format = 'png')
        print('Saved contour_perit_'+tag+str(idx)+' plot at ', save_path)


    return