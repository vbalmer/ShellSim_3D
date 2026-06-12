# -------------------------------------------------------------------------------------------------------------------- #
# Input Definition and Meshing
# (C) Andreas Näsbom, ETH Zürich
# 20.08.2021
# -------------------------------------------------------------------------------------------------------------------- #

# print("-------------------------------------------------------")
# print("1 Start Meshing with gmsh")
# print("-------------------------------------------------------")
# print("1.1 Assemble Model and mesh")
import gmsh
import numpy as np
import math
from math import *
from numpy import dot
from numpy.linalg import norm

# -------------------------------------------------------------------------------------------------------------------- #
# 0 Auxiliary Functions
# -------------------------------------------------------------------------------------------------------------------- #

def gauss_points(nn,go):
    if nn == 4:
        if go == 1:
            xi = np.array([0])
            w = np.array([2])
        elif go == 2:
            xi = np.array([-sqrt(1/3),sqrt(1/3)])
            w = np.array([1,1])
        elif go == 3:
            xi = np.array([-0.774597, 0, 0.774597])
            w = np.array([5/9,8/9,5/9])
        elif go == 4:
            xi = np.array([-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053])
            w = np.array([0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538])
        elif go == 5:
            xi = np.array([-0.9061798459386640,-0.5384693101056831,0,0.5384693101056831,0.9061798459386640])
            w = np.array([0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891])
        elif go == 6:
            xi1 = -0.9324695142031521
            xi2 = -0.6612093864662645
            xi3 = -0.2386191860831969
            xi4 = 0.2386191860831969
            xi5 = 0.661209386466264
            xi6 = 0.9324695142031521
            a1 = 0.1713244923791704
            a2 = 0.3607615730481387
            a3 = 0.4679139345726913
            a4 = a3
            a5 = a2
            a6 = a1
            xi = np.array([xi1,xi2,xi3,xi4,xi5,xi6])
            w = np.array([a1,a2,a3,a4,a5,a6])
        elif go == 30:
            w = [0.102852653,0.102852653,0.10176239,0.10176239,0.099593421,0.099593421,0.096368737,0.096368737,0.092122522,0.092122522,
                 0.086899787,0.086899787,0.080755895,0.080755895,0.073755975,0.073755975,0.06597423,0.06597423,0.057493156,0.057493156,
                 0.048402673,0.048402673,0.038799193,0.038799193,0.028784708,0.028784708,0.018466468,0.018466468,0.007968192,0.007968192]
            xi = [-0.051471843,0.051471843,-0.153869914,0.153869914,-0.254636926,0.254636926,-0.352704726,0.352704726,-0.44703377,0.44703377,
                  -0.536624148,0.536624148,-0.620526183,0.620526183,-0.697850495,0.697850495,-0.767777432,0.767777432,-0.829565762,0.829565762,
                  -0.882560536,0.882560536,-0.926200047,0.926200047,-0.960021865,0.960021865,-0.983668123,0.983668123,-0.996893484,0.996893484]
        elif go == 40:
            w = [0.077505948,
                0.077505948,
                0.077039818,
                0.077039818,
                0.076110362,
                0.076110362,
                0.074723169,
                0.074723169,
                0.072886582,
                0.072886582,
                0.070611647,
                0.070611647,
                0.067912046,
                0.067912046,
                0.064804013,
                0.064804013,
                0.061306242,
                0.061306242,
                0.057439769,
                0.057439769,
                0.053227847,
                0.053227847,
                0.048695808,
                0.048695808,
                0.043870908,
                0.043870908,
                0.038782168,
                0.038782168,
                0.033460195,
                0.033460195,
                0.027937007,
                0.027937007,
                0.022245849,
                0.022245849,
                0.016421058,
                0.016421058,
                0.010498285,
                0.010498285,
                0.004521277,
                0.004521277]
            xi = [-0.038772418,
            0.038772418,
            -0.116084071,
            0.116084071,
            -0.192697581,
            0.192697581,
            -0.268152185,
            0.268152185,
            -0.341994091,
            0.341994091,
            -0.413779204,
            0.413779204,
            -0.483075802,
            0.483075802,
            -0.549467125,
            0.549467125,
            -0.61255389,
            0.61255389,
            -0.671956685,
            0.671956685,
            -0.727318255,
            0.727318255,
            -0.778305651,
            0.778305651,
            -0.824612231,
            0.824612231,
            -0.865959503,
            0.865959503,
            -0.902098807,
            0.902098807,
            -0.932812808,
            0.932812808,
            -0.957916819,
            0.957916819,
            -0.97725995,
            0.97725995,
            -0.990726239,
            0.990726239,
            -0.99823771,
            0.99823771]
        elif go == 50:
            w = [0.062176617,
                0.062176617,
                0.061936067,
                0.061936067,
                0.0614559,
                0.0614559,
                0.060737971,
                0.060737971,
                0.059785059,
                0.059785059,
                0.05860085,
                0.05860085,
                0.057189926,
                0.057189926,
                0.055557745,
                0.055557745,
                0.053710622,
                0.053710622,
                0.051655703,
                0.051655703,
                0.049400938,
                0.049400938,
                0.046955051,
                0.046955051,
                0.044327504,
                0.044327504,
                0.041528463,
                0.041528463,
                0.038568757,
                0.038568757,
                0.035459836,
                0.035459836,
                0.032213728,
                0.032213728,
                0.028842994,
                0.028842994,
                0.025360674,
                0.025360674,
                0.021780243,
                0.021780243,
                0.018115561,
                0.018115561,
                0.014380823,
                0.014380823,
                0.010590548,
                0.010590548,
                0.006759799,
                0.006759799,
                0.002908623,
                0.002908623]
            xi = [-0.031098338,
                0.031098338,
                -0.093174702,
                0.093174702,
                -0.15489059,
                0.15489059,
                -0.216007237,
                0.216007237,
                -0.276288194,
                0.276288194,
                -0.335500245,
                0.335500245,
                -0.393414312,
                0.393414312,
                -0.449806335,
                0.449806335,
                -0.504458145,
                0.504458145,
                -0.557158305,
                0.557158305,
                -0.607702927,
                0.607702927,
                -0.655896466,
                0.655896466,
                -0.701552469,
                0.701552469,
                -0.744494302,
                0.744494302,
                -0.784555833,
                0.784555833,
                -0.821582071,
                0.821582071,
                -0.855429769,
                0.855429769,
                -0.88596798,
                0.88596798,
                -0.913078557,
                0.913078557,
                -0.936656619,
                0.936656619,
                -0.956610955,
                0.956610955,
                -0.972864385,
                0.972864385,
                -0.985354084,
                0.985354084,
                -0.994031969,
                0.994031969,
                -0.998866404,
                0.998866404]
    elif nn == 3:
        if go == 1:
            xi = np.array([1/3])
            w = np.array([1 / sqrt(2)])
        elif go == 2:
            xi = np.array([1/6,2/3])
            w = np.array([1/sqrt(6),1/sqrt(6)])
        # if go == 1:
        #     xi = np.array([0])
        #     w = np.array([2])
        # if go == 2:
        #     xi = np.array([1/3-1,4/3-1])
        #     w = np.array([1/sqrt(6),1/sqrt(6)])
    return xi,w


def jacobi (k,i,j,go,NODES,ELEMENTS):
    # Node Connectivity of analyzed element
    ek = ELEMENTS[k,:][ELEMENTS[k,:] < 10**5]

    # -----------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------ Integration of quadrilaterals ----------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    if len(ek) == 4:
        # Coordinates of element nodes
        v = np.array(NODES[ELEMENTS[k]])

        # Gauss Points and weights
        gp, w = gauss_points(4,go)
        xi = gp[j]
        eta = gp[i]

        # Shape Functions in xi-eta space
        N1xe = 1/4*(1-xi)*(1-eta)
        N2xe = 1/4*(1+xi)*(1-eta)
        N3xe = 1/4*(1+xi)*(1+eta)
        N4xe = 1/4*(1-xi)*(1+eta)

        # Derivations Shape Functions in xi-eta space
        N1xi = float(- (1 - eta) / 4)
        N2xi = float((1 - eta) / 4)
        N3xi = float((1 + eta) / 4)
        N4xi = float(- (1 + eta) / 4)
        N1eta = float(- (1 - xi) / 4)
        N2eta = float(- (1 + xi) / 4)
        N3eta = float((1 + xi) / 4)
        N4eta = float((1 - xi) / 4)

        # Gradient Matrix
        Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

        # Jacobian
        J = np.matmul(Grad_Mat, v)
        J_inv = np.linalg.inv(J)
        J_det = np.linalg.det(J)
    # -----------------------------------------------------------------------------------------------------------------#
    # -------------------------------------------- Integration of triangles -------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    else:
        # Coordinates of element nodes
        v = np.array(NODES[ek])
        # Gauss Points and weights
        gp, w = gauss_points(3, go)
        xi = gp[j]
        eta = gp[i]-1/3

        # Shape Functions in xi-eta space
        # N1xe = 1-xi-eta
        # N2xe = xi
        # N3xe = eta
        N1xe = 1 / 4 * (1 - xi) * (1 - eta)
        N2xe = 1 / 4 * (1 + xi) * (1 - eta)
        N3xe = 1 / 2 * (1 + eta)

        # Derivations Shape Functions in xi-eta space
        # N1xi = -1
        # N2xi = 1
        # N3xi = 0
        # N1eta = -1
        # N2eta = 0
        # N3eta = 1
        N1xi = float(- (1 - eta) / 4)
        N2xi = float((1 - eta) / 4)
        N3xi = 0
        N1eta = float(- (1 - xi) / 4)
        N2eta = float(- (1 + xi) / 4)
        N3eta = 1/2


        # Gradient Matrix
        Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

        # Jacobian
        J = np.matmul(Grad_Mat, v)
        J_inv = np.linalg.inv(J)
        J_det = np.linalg.det(J)

    return xi,eta,J,J_inv,J_det


def vcos(a,b):
    cos_ab = dot(a,b) / (norm(a) * norm(b))
    return cos_ab


def find_copl(n, ELS, Tk1):
    copl = 1
    rowsk = np.where(ELS[0] == n)
    rowsk = rowsk[0].flatten()
    for k in range(len(rowsk)):
        k_k = rowsk[k]
        if k == 0:
            Tk1_ref = Tk1[k_k]
        else:
            Tk1_k = Tk1[k_k]
            diff_tk1 = Tk1_ref-Tk1_k
            if abs(np.max(np.max(diff_tk1)))>10**-5:
                copl = 0
    return copl


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def length(v):
  return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))



'''
" -------------------------------------------------------------------------------------------------------------------- "
" -------------------------------------------------------------------------------------------------------------------- "
" Plate examples "
" -------------------------------------------------------------------------------------------------------------------- "
" -------------------------------------------------------------------------------------------------------------------- "

    # -------------------------------------------------------------------------------------------------------------------- #
    # 0 Geometry Input
    # -------------------------------------------------------------------------------------------------------------------- #
'''
def input_definition(mat, NN_hybrid):
    '''
    ---------------------------------------------------
    Definition of geometry and boundary conditions: 
    1. Geometry and Mesh
    2. Loads & BCs
    3. Material & further parameters
    4. Postprocess Mesh
    ---------------------------------------------------

    Inputs: 
    mat     (dict)          Containing relevant parameters

    Outputs (read by main.py file):
    MATK
    NODESG
    ELS
    COORD
    GEOMA
    GEOMK
    MASK
    na
    BC
    Tk1

    '''
    print("-------------------------------------------------------")
    print("1 Start Meshing with gmsh")
    print("-------------------------------------------------------")
    print("1.1 Assemble Model and mesh")
    
    # unpacking dict

    for key in ['L', 'B', 'E_1', 'E_2', 'ms', 'F_N', 's', 't_1', 't_2', 'nl', 'nu_1', 'nu_2', 'mat', 'rho_x', 'rho_y']: 
        if isinstance(mat[key], np.ndarray):
            mat[key] = mat[key][0]

    L = mat['L']
    B = mat['B']
    E_young_1 = mat['E_1']
    E_young_2 = mat['E_2']
    ms = mat['ms']
    Force_mag = mat['F']
    F_N = mat['F_N']
    scenario = mat['s']
    thickness_1 = mat['t_1']
    thickness_2 = mat['t_2']
    n_layer = mat['nl']
    nu_1 = mat['nu_1']
    nu_2 = mat['nu_2']
    mat_type = mat['mat']

    

    if mat_type == 3: 
        # unpack additional values
        tb0 = mat['tb0']
        tb1 = mat['tb1']
        ect = mat['ect']
        ec0 = mat['ec0']
        fcp = mat['fcp']
        fct = mat['fct']
        fsy = mat['fsy']
        fsu = mat['fsu']
        Es = mat['Es']
        Esh = mat['Esh']
        rho_x = mat['rho_x']
        rho_y = mat['rho_y']
        D = mat['D']
    elif mat_type == 1 or mat_type == 10:
        # these numbers won't be used but need to be defined for the program to work
        tb0 = 5.8
        tb1 = 2.9
        ect = 0.09e-3
        ec0 = 2.3e-3
        fcp = 30
        fct = 2.9
        fsy = 435
        fsu = 500
        Es = 205e3
        Esh = 8e3
        rho_x = 0.01
        rho_y = 0.01
        D = 16

    # -------------------------------------------------------------------------------------------------------------------- #
    # 1 Define Geometry and Mesh with gmsh
    # -------------------------------------------------------------------------------------------------------------------- #

    # 1.1 Initialize
    " 1.1 Output: gmsh.model "
    gmsh.initialize()
    gmsh.model.add("t11")

    # 1.2 Define Points
    " 1.2 Output:   - pi: Points in gmsh.model.geo(x, y, z) "
    "               - ms: Mesh size"
    # ms=L/2
    gmsh.model.geo.addPoint(0,0,0,ms)
    gmsh.model.geo.addPoint(L,0,0,ms)
    gmsh.model.geo.addPoint(L,B,0,ms)
    gmsh.model.geo.addPoint(0,B,0,ms)

    # 1.3 Define Lines
    " 1.3 Output:   - li: Lines in gmsh.model.geo() "
    gmsh.model.geo.addLine(1,2)
    gmsh.model.geo.addLine(2,3)
    gmsh.model.geo.addLine(3,4)
    gmsh.model.geo.addLine(4,1)

    # 1.4 Define Areas
    " 1.4 Output:   - cl: Areas in gmsh.model.geo() "
    na = 1
    cl = [[]]*na
    cl[0] = gmsh.model.geo.addCurveLoop([1,2,3,4])

    # 1.5 Origins of Local Coordinate Systems in Global Coordinates
    " 1.5 Output:   - O: Origins of areas in global coordinate systems "
    O=[[]*3]*na
    O[0] = [0,0,0]

    # 1.6 Points defining local x- and y coordinate systems: [Origin,x,y]
    " 1.6 Output:   - LS: global points defining local x - and y - axes of areas "
    LS=[[]]*na
    LS[0] = np.array([[0,0,0],[1,0,0],[0,1,0]])

    # 1.7 Miscellaneous information
    " 1.7 Output:   - mesh: Approximate mesh size in local x - and y - directions per area (needed for plot) "
    "               - nl: Number of layers per area"
    mesh = [[ms,ms]]
    nl = [int(n_layer)]*na

    # 1.8 Assembly into gmsh Model and Embedded Lines
    " 1.8 Output:   - model: Embedded lines in areas to provide connectivity"
    "               - model/pl/field: Assembly into gmsh.model() "
    for i in range(na):
        pl = gmsh.model.geo.addPlaneSurface([cl[i]])
        gmsh.model.geo.synchronize()

    field = gmsh.model.mesh.field

    # 1.9 Perform meshing
    " 1.9 Output:   - Mesh "
    # - RecombineAll = 0:    triangular elements
    # - RecombineAll = 1:    recombination into quads
    # - Mesh.Algorithm = 1:  MeshAdapt
    # - Mesh.Algorithm = 2:  Automatic
    # - Mesh.Algorithm = 3:  Initial mesh only
    # - Mesh.Algorithm = 5:  Delauny
    # - Mesh.Algorithm = 6:  Frontal-Delauny
    # - Mesh.Algorithm = 7:  BAMG
    # - Mesh.Algorithm = 8:  frontal-delauny for quads
    # - Mesh.Algorithm = 9:  Packing of Parallelograms
    gmsh.option.setNumber("Mesh.RecombineAll",1)
    gmsh.option.setNumber("Mesh.Algorithm",8)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm",1)
    gmsh.model.mesh.generate(2)

    # 1.10 Launch the gmsh GUI to See the Results:
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    # -------------------------------------------------------------------------------------------------------------------- #
    # 2 Loads and Boundary Conditions
    # -------------------------------------------------------------------------------------------------------------------- #
    print("1.2 Boundary Conditions and Loads")
    # 2.1 Boundary Conditions
    
    ################ in-plane base cases ################

    if scenario == 8: 
        # pure membrane shear
        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))


        # 2.2 Loads
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])

            Load_n[i]  = np.array([
                            [0+ms/2,L-ms/2,0,0,0,0,1,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,1,Force_mag[i]*(ms/L)],
                            [0,0,0+ms/2,B-ms/2,0,0,2,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,2,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            ])

    elif scenario == 9: 
        # pure tension / compression in x-direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))



        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([
                            # tension in x-direction
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # minor compression in y-direction
                            [0+ms/2,L-ms/2,0,0,0,0,2,0.18e6*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,-0.18e6*(ms/L)],

                            [0,0,0,0,0,0,2,0.18e6/2*(ms/L)],
                            [L,L,0,0,0,0,2,0.18e6/2*(ms/L)],
                            [0,0,B,B,0,0,2,-0.18e6/2*(ms/L)],
                            [L,L,B,B,0,0,2,-0.18e6/2*(ms/L)],
                            ])
        
    elif scenario == 109: 
        # pure tension / compression in y-direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"


        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads in y-direction
            Load_n[i]  = np.array([
                            # tension / compression in y-direction
                            [0+ms/2,L-ms/2,0,0,0,0,2,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],


                            # minor compression in x-direction
                            [0,0,0+ms/2,B-ms/2,0,0,1,180e3*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,-180e3*(ms/L)],

                            [0,0,0,0,0,0,1,180e3/2*(ms/L)],
                            [L,L,0,0,0,0,1,-180e3/2*(ms/L)],
                            [0,0,B,B,0,0,1,180e3/2*(ms/L)],
                            [L,L,B,B,0,0,1,-180e3/2*(ms/L)],


                            ])

    ################ in-plane combined cases ################

    elif scenario == 110:
        # pure tension /compression in x and y direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads in y-direction
            Load_n[i]  = np.array([
                            # tension / compression in x-direction on all sides
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],

                            # tension / compression in y-direction on all sides
                            [0+ms/2,L-ms/2,0,0,0,0,2,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,Force_mag[i]*(ms/L)],

                            # tension / compression in x-direction at edges
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # tension / compression in y-direction at edges
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            ])
        
    elif scenario == 111:
        # tension in x-direction and compression y-direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([
                            # tension at sides in x-direction
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],
                            
                            # compression at other sides in y-direction
                            [0+ms/2,L-ms/2,0,0,0,0,2,Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,-Force_mag[i]*(ms/L)],

                            # tension at edges in x-direction
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # compression at edges in y-direction
                            [0,0,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            ])
        
    elif scenario == 112:
        # tension/compression in x-direction and shear

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads in y-direction
            Load_n[i]  = np.array([
                            # tension / compression in x-direction
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            #shear
                            [0+ms/2,L-ms/2,0,0,0,0,1,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,1,Force_mag[i]*(ms/L)],
                            [0,0,0+ms/2,B-ms/2,0,0,2,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,2,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            ])
        
    elif scenario == 113:
    # pure compression in x and y direction + shear

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        
        
        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                        [0,L,0,B,0,0,3,0],
                        ])

            # nodal loads in y-direction
            Load_n[i]  = np.array([
                            # tension / compression in x-direction on all sides
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],

                            # tension / compression in y-direction on all sides
                            [0+ms/2,L-ms/2,0,0,0,0,2,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,Force_mag[i]*(ms/L)],

                            # tension / compression in x-direction at edges
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # tension / compression in y-direction at edges
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],

                            # shear on all sides
                            [0+ms/2,L-ms/2,0,0,0,0,1,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,1,Force_mag[i]*(ms/L)],
                            [0,0,0+ms/2,B-ms/2,0,0,2,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,2,Force_mag[i]*(ms/L)],

                            # shear at edges
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            ])
             
    elif scenario == 114:
        # tension in x-direction and compression y-direction + shear

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([
                            # tension at sides in x-direction
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],
                            
                            # compression at other sides in y-direction
                            [0+ms/2,L-ms/2,0,0,0,0,2,Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,-Force_mag[i]*(ms/L)],

                            # tension at edges in x-direction
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # compression at edges in y-direction
                            [0,0,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
            
                            # shear on all sides
                            [0+ms/2,L-ms/2,0,0,0,0,1,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,1,Force_mag[i]*(ms/L)],
                            [0,0,0+ms/2,B-ms/2,0,0,2,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,2,Force_mag[i]*(ms/L)],

                            # shear at edges
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            ])

    elif scenario == 115:
    # pure compression in x and y direction + shear

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,0,0,0,0],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,0,0,0],
                        ]))
        
        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"

        
        
        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                        [0,L,0,B,0,0,3,0],
                        ])

            # nodal loads in y-direction
            Load_n[i]  = np.array([
                            # tension / compression in x-direction on all sides
                            [0,0,0+ms/2,B-ms/2,0,0,1,-Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,1,Force_mag[i]*(ms/L)],

                            # tension / compression in y-direction on all sides
                            [0+ms/2,L-ms/2,0,0,0,0,2,-Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,2,Force_mag[i]*(ms/L)],

                            # tension / compression in x-direction at edges
                            [0,0,0,0,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,-Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,Force_mag[i]/2*(ms/L)],

                            # tension / compression in y-direction at edges
                            [0,0,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,-Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,Force_mag[i]/2*(ms/L)],

                            # shear on all sides, but 2 times larger than nx and ny
                            [0+ms/2,L-ms/2,0,0,0,0,1,-2*Force_mag[i]*(ms/L)],
                            [0+ms/2,L-ms/2,B,B,0,0,1,2*Force_mag[i]*(ms/L)],
                            [0,0,0+ms/2,B-ms/2,0,0,2,-2*Force_mag[i]*(ms/L)],
                            [L,L,0+ms/2,B-ms/2,0,0,2,2*Force_mag[i]*(ms/L)],

                            # shear at edges
                            [0,0,0,0,0,0,1,-2*Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,1,-2*Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,1,2*Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,1,2*Force_mag[i]/2*(ms/L)],
                            [0,0,0,0,0,0,2,-2*Force_mag[i]/2*(ms/L)],
                            [0,0,B,B,0,0,2,-2*Force_mag[i]/2*(ms/L)],
                            [L,L,0,0,0,0,2,2*Force_mag[i]/2*(ms/L)],
                            [L,L,B,B,0,0,2,2*Force_mag[i]/2*(ms/L)],
                            ])


    ################ out-of-plane base cases ################

    elif scenario == 201: 
        # pure bending in y-direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,1234,1234,1234],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,1234,1234,1234,1234],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,1234,1234,1234],
                        [-f, f, B-f, B+f, -f, f, 1234, 1234,0,1234,1234,1234],
                        ]))

        
        # BC = np.array(([
        #                 [L/2-f, L/2+f, B/2-f, B/2+f, -f, f, 0,0,0,0,0,0],
        #                 ]))

        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"


        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([
                            # minor negative bending in y-direction
                            # [0,0,0+ms/2,L-ms/2,0,0,5,180e3*(ms/L)],
                            # [B,B,0+ms/2,L-ms/2,0,0,5,-180e3*(ms/L)],

                            # [0,0,0,0,0,0,5,180e3/2*(ms/L)],
                            # [0,0,L,L,0,0,5,-180e3/2*(ms/L)],
                            # [B,B,0,0,0,0,5,180e3/2*(ms/L)],
                            # [B,B,L,L,0,0,5,-180e3/2*(ms/L)],


                            # bending in x-direction
                            [0+ms/2,B-ms/2,0,0,0,0,4,-Force_mag[i]*(ms/L)],
                            [0+ms/2,B-ms/2,L,L,0,0,4,Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,4,-Force_mag[i]/2*(ms/L)],   # (0,0)  y=0  -> -F/2
                            [0,0,L,L,0,0,4,Force_mag[i]/2*(ms/L)],    # (0,L)  y=L  -> +F/2
                            [B,B,0,0,0,0,4,-Force_mag[i]/2*(ms/L)],   # (B,0)  y=0  -> -F/2
                            [B,B,L,L,0,0,4,Force_mag[i]/2*(ms/L)],    # (B,L)  y=L  -> +F/2


                            ])
        
    elif scenario == 202: 
        # pure bending in x-direction

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        BC = np.array(([
                        [-f, f, -f, f, -f, f, 1234, 0,0,1234,1234,1234],
                        [L-f, L+f, -f, f, -f, f, 1234, 0,1234,1234,1234,1234],
                        [L-f, L+f, B-f, B+f, -f, f, 0, 1234,0,1234,1234,1234],
                        [-f, f, B-f, B+f, -f, f, 1234, 1234,0,1234,1234,1234],
                        ]))

        
        # BC = np.array(([
        #                 [L/2-f, L/2+f, B/2-f, B/2+f, -f, f, 0,0,0,0,0,0],
        #                 ]))

        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"


        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([
                            # minor negative bending in y-direction
                            # [0,0,0+ms/2,L-ms/2,0,0,5,180e3*(ms/L)],
                            # [B,B,0+ms/2,L-ms/2,0,0,5,-180e3*(ms/L)],

                            # [0,0,0,0,0,0,5,180e3/2*(ms/L)],
                            # [0,0,L,L,0,0,5,-180e3/2*(ms/L)],
                            # [B,B,0,0,0,0,5,180e3/2*(ms/L)],
                            # [B,B,L,L,0,0,5,-180e3/2*(ms/L)],


                            # bending in x-direction
                            [0,0,0+ms/2,L-ms/2,0,0,5,Force_mag[i]*(ms/L)],
                            [B,B,0+ms/2,B-ms/2,0,0,5,-Force_mag[i]*(ms/L)],

                            [0,0,0,0,0,0,5,Force_mag[i]/2*(ms/L)],   # (0,0)  x=0  -> F/2
                            [B,B,0,0,0,0,5,-Force_mag[i]/2*(ms/L)],    # (B,0)  x=B  -> -F/2
                            [0,0,L,L,0,0,5,Force_mag[i]/2*(ms/L)],   # (0,L)  x=0  -> +F/2
                            [B,B,L,L,0,0,5,-Force_mag[i]/2*(ms/L)],    # (B,L)  x=B  -> -F/2


                            ])

    elif scenario == 203: 
        # pure twisting

        " 2.1 Output:    - Global Boundary Conditions "
        "                  [xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]"
        "                   BC_i in unit length (mm), 1234 if DOF is free"

        
        f = 1
        
        # BC = np.array(([
        #                 [L/2-f, L/2+f, B/2-f, B/2+f, -f, f, 0,0,0,0,0,0],
        #                 ]))

        BC = np.array(([
                        [L/2-f, L/2+f, -f, f, -f, f, 1234, 0,0,1234,1234,1234],
                        [L/2-f, L/2+f, B-f, B+f, -f, f, 0, 1234,0,1234,1234,1234],
                        [L-f, L+f, B/2-f, B/2+f, -f, f, 0, 1234,0,1234,1234,1234],
                        # [-f, f, B/2-f, B/2+f, -f, f, 1234, 0,0,1234,1234,1234],
                        ]))

        
        " 2.2 Output:   - Load_el: Global Element Loads "
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]]"
        "               - Load_n: Global Nodal Loads"
        "                 [xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]]"


        if isinstance(Force_mag, np.ndarray): 
            nls = len(Force_mag)
            Load_el = [[]]*nls
            Load_n = [[]]*nls
        else: 
            nls = 1
            Force_mag = [Force_mag]
            Load_el = [[]]*nls
            Load_n = [[]]*nls

        for i in range(nls):
            # no element loads (magnitude = 0)
            Load_el[i] = np.array([
                            [0,L,0,B,0,0,3,0],
                            ])
            
            # nodal loads
            Load_n[i]  = np.array([

                            #twisting along the y-edges
                            [0,0,0+ms/2,L-ms/2,0,0,4,Force_mag[i]*(ms/L)],
                            [B,B,0+ms/2,B-ms/2,0,0,4,-Force_mag[i]*(ms/L)],
                            #twisting along the x-edges
                            [0+ms/2,L-ms/2,0,0,0,0,5,-Force_mag[i]*(ms/L)],
                            [0+ms/2,B-ms/2,B,B,0,0,5,Force_mag[i]*(ms/L)],

                            # in x-direction at edges
                            [0,0,0,0,0,0,4,Force_mag[i]/2*(ms/L)],   # (0,0)  x=0  -> F/2
                            [B,B,0,0,0,0,4,-Force_mag[i]/2*(ms/L)],    # (B,0)  x=B  -> -F/2
                            [0,0,L,L,0,0,4,Force_mag[i]/2*(ms/L)],   # (0,L)  x=0  -> +F/2
                            [B,B,L,L,0,0,4,-Force_mag[i]/2*(ms/L)],    # (B,L)  x=B  -> -F/2


                            # in y-direction at edges
                            [0,0,0,0,0,0,5,-Force_mag[i]/2*(ms/L)],   # (0,0)  x=0  -> -F/2
                            [B,B,0,0,0,0,5,-Force_mag[i]/2*(ms/L)],    # (B,0)  x=B  -> -F/2
                            [0,0,L,L,0,0,5,Force_mag[i]/2*(ms/L)],   # (0,L)  x=0  -> +F/2
                            [B,B,L,L,0,0,5,Force_mag[i]/2*(ms/L)],    # (B,L)  x=B  -> +F/2

                            ])



    # -------------------------------------------------------------------------------------------------------------------- #
    # 3 Define Material and further Parameters
    # -------------------------------------------------------------------------------------------------------------------- #
    print("1.3 Read Material, Constitutive Model and Integration Options")

    # 3.1 Constitutive Model
    " 3.1 Output:   - cma: Applied Constitutive Model for each area"
    # - 1 = Linear Elastic
    # - 3 = CMM
    cma = [mat_type]

    # 3.2 Iteration Type
    " 3.2 Output:   - it_type"
    "                   it_type == 1: Tangent stiffness iteration (Newton-Raphson)"
    "                   it_type == 2: Secant stiffness iteration (Fixed-Point)"
    it_type = 1

    # 3.3 Shape Function Order
    " 3.3 Output:   - order: Polynomial Order of Shape Functions"
    # - 1 = Linear Shape Functions
    # - 2 = Quadratic Shape Functions (NOT YET IMPLEMENTED)
    order = 1

    # 3.4 Gauss Order
    " 3.4 Output:   - gauss_order: Number of Gauss Points per element in bending and shear"
    if NN_hybrid['predict_sig'] or NN_hybrid['predict_D']: 
        gauss_order = 2
    else: 
        gauss_order = 2

    # 3.5 Material Properties / Long Term Properties
    " 3.5 Output:   - Prop[ia]: Material and reinforcement properties per area"
    "               - GEOMA: Geometry information per area"
    "               - MATA: Material information per area"
    # - Ec:     E-Modulus of Concrete               [MPa]
    # - Gc:     Shear Modulus of Concrete           [MPa]
    # - vc:     Poisson's Ratio of Concrete         [-]
    # - t:      Thickness of Area                   [mm]
    # - fcp:    Concrete Strength                   [PMa]
    # - fct:    Concrete Tensile Strength           [MPa]
    # - tb0:    Bond Shear Stress for Elastic Steel [MPa]
    # - tb1:    Bond Shear Stress for Plastic Steel [MPa]
    # - ec0:    Concrete Strain at Peak Stress      [-]
    # - rhox:   Reinforcement Content in x per Layer[-]
    # - dx:     Reinforcement Bar Diameter in x     [mm]
    # - Esx:    E-Modulus of Steel in x             [MPa]
    # - Eshx:   Hardening Modulus of Steel in x     [MPa]
    # - fsyx:   Yield Stress of Steel in x          [MPa]
    # - fsux:   Ultimate Stress of Steel in x       [MPa]
    # - phi:    creep coefficient                   [-]
    #           not implemented yet
    # - ecs:    free shrinkage strain               [-]
    #           only applicable in LFEA
    # - Subscripts y: Same Procedure in y - Direction
    # - IMPORTANT: x and y are local area coordinate systems

    E_y1 = [E_young_1]*na
    E_y2 = [E_young_2]*na
    Gc = [E_young_1/(2*(1+0.2))] * na
    vc = [nu_1] * na # @bav: Bei 0 lassen für Nichtlinearen Fall, da sonst Konvergenzprobleme (lin. funktioniert)
    vc_2 = [nu_2] * na
    t = [thickness_1]*na
    t_2 = [thickness_2] *na
    fcp = [fcp] * na
    fct = [fct] * na
    tb0 = [tb0] * na
    tb1 = [tb1] * na
    ec0 = [ec0] * na
    Dmax = [16] * na
    rhox = [[rho_x,rho_x,rho_x,rho_x,0,0,0,0,0,0,0,0,0,0,0,0,rho_x,rho_x,rho_x,rho_x]]*na
    dx = [[D]*nl[0]]
    sx = [[200]*nl[0]]
    Esx = [Es]*na
    Eshx = [Esh] * na
    fsyx = [fsy]
    fsux = [fsu]
    rhoy =[[rho_y,rho_y,rho_y,rho_y,0,0,0,0,0,0,0,0,0,0,0,0,rho_y,rho_y,rho_y,rho_y]]*na
    dy  = [[D]*nl[0]] * na
    sy = [[200]*nl[0]]
    Esy = [Es]*na
    Eshy = [Esh] * na
    fsyy = [fsy] * na
    fsuy = [fsu] * na


    # @bav: folgende parameter von phi bis fpuy unwichtig für dich (alle rhop müssen = 0 sein, a ansonsten CFK vorspannung)
    phi = [0] * na
    ecs = [-0.000] * na

    tbp0 = [10] * na
    tbp1 = [8] * na
    ebp1 = [1e-2]*na
    rhopx = [[0]*nl[0]]
    dpx  = [[8.2]*nl[0]]*na
    Epx = [149000] * na
    fpux = [1700] * na
    rhopy = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]*na
    dpy  = [[8.2]*nl[0]]*na
    Epy = [149000] * na
    fpuy = [1700] * na
    # Output

    GEOMA ={"na"            : na,
            "nla"           : nl,
            "Oa"            : O,
            "ta"            : t,
            "ta_2"          : t_2,
            "meshsa"        : mesh,
            "rhoxal"        : rhox,
            "dxal"          : dx,
            "sxal"          : sx,
            "rhoyal"        : rhoy,
            "dyal"          : dy,
            "syal"          : sy,
            "rhopxal"       : rhopx,
            "dpxal"         : dpx,
            "rhopyal"       : rhopy,
            "dpyal"         : dpy
            }
    MATA = {"Eca"           : E_y1,
            "Eca2"          : E_y2,
            "Gca"           : Gc,
            "vca"           : vc,
            "vca2"          : vc_2,
            "fcpa"          : fcp,
            "fcta"          : fct,
            "tb0a"          : tb0,
            "tb1a"          : tb1,
            "tbp0a"         : tbp0,
            "tbp1a"         : tbp1,
            "ebp1a"         : ebp1,
            "ec0a"          : ec0,
            "Dmaxa"         : Dmax,
            "Esxa"          : Esx,
            "Eshxa"         : Eshx,
            "fsyxa"         : fsyx,
            "fsuxa"         : fsux,
            "Esya"          : Esy,
            "Eshya"         : Eshy,
            "fsyya"         : fsyy,
            "fsuya"         : fsuy,
            "Epxa"          : Epx,
            "Epya"          : Epy,
            "fpuxa"         : fpux,
            "fpuya"         : fpuy,
            "cma"           : cma,
            "phia"          : phi,
            "ecsa"          : ecs,}



    '-----------------------------------------------------------------------------------------------------------------------'
    # -------------------------------------------------------------------------------------------------------------------- #
    # 4 Postprocess Mesh
    # -------------------------------------------------------------------------------------------------------------------- #
    from numpy import random
    print("1.4 Postprocess Mesh")
    # 4.0 Initiation of Output
    ELS = [[],[[]]*na,[[]]*na,[[]]*na,[]]
    COORD = {}
    GEOMA["T1"]=['']*na
    GEOMA["T2"]=['']*na
    GEOMA["T3"]=['']*na

    # 4.1 Coordinates of Nodes
    " 4.1 Output:   - NODESG: Global Coordinates of all Nodes "
    print("   1.41 Node Coordinates")
    a = gmsh.model.mesh.get_nodes()
    nn = int(len(a[1])/3)
    NODESG = np.zeros((nn,3))
    NODESG[:,0]=a[1][0::3]
    NODESG[:,1]=a[1][1::3]
    NODESG[:,2]=a[1][2::3]

    lenn = len(NODESG[:,0])
    # NODESG[:,0] = NODESG[:,0]+random.rand(lenn)*0
    # NODESG[:,1] = NODESG[:,1]+random.rand(lenn)*0
    # 4.2 Element Connectivity
    " 4.2 Output:   - tris & quads: triangles and quadrilaterals and their defining node indices (starting at 0)"
    print("   1.42 Node Connectivity")
    b=gmsh.model.mesh.get_elements()
    if 3 in b[0]:
        if 2 in b[0]:
            tris = b[2][1]
            tris = tris.reshape((int(len(tris) / 3)), 3)
            quads = b[2][2]
            quads = quads.reshape((int(len(quads) / 4)), 4)
        else:
            tris = [[]]
            quads = b[2][1]
            quads = quads.reshape((int(len(quads)/4)),4)
    else:
        quads = [[]]
        tris = b[2][1]
        tris = tris.reshape((int(len(tris) / 3)), 3)
    if len(tris[0]) > 1 and len(quads[0]) > 1:
        nk = len(quads[:,0])+len(tris[:,0])
        tri_indices = b[1][1]
        quad_indices = b[1][2]
    elif len(tris[0])<1:
        nk = len(quads[:,0])
        tri_indices = []
        quad_indices = b[1][1]
    elif len(quads[0])<1:
        nk = len(tris[:,0])
        quad_indices = []
        tri_indices = b[1][1]
    quads = quads - np.ones_like(quads)
    tris = tris - np.ones_like(tris)

    # 4.3 Element Center Point Information (1/2), see 4.8
    " 4.3 Output:   - centersG: Element Center Points in Global Coordinates"
    "               - el_surf: Area (Surface) ID of Elements"
    print("   1.43 Element Center Information 1/2")
    el_surf = np.zeros(nk)
    for ia in range(1,na+1):
        for iic in range(2):
            type = [3,2][iic]
            c_ia = gmsh.model.mesh.get_barycenters(type,ia,0,1)
            c_ia = c_ia.reshape((int(len(c_ia)/3)),3)
            if ia == 1 and iic == 0:
                centersG = c_ia
                el_surf = np.zeros((len(c_ia[:,0])))
            else:
                centersG = np.append(centersG,c_ia,axis=0)
                el_surf = np.append(el_surf,(ia-1)*np.ones((len(c_ia[:,0]),1)))
    el_surf = el_surf.astype(int)

    # 4.4 Assign Element Numbers according to center numbers
    " 4.4 Output:   - els_sort: Elements sorted by centersG"
    "               - type_sort: Nodes per Element: 3 for tri & 4 for quad"
    "               - ELS[0]: els_sort"
    "               - ELS[1]: type_sort"
    # - Order: quads of surface 1, tris of surface 1, quads of surf 2, etc.
    # - Triangular elements get node 100'001 as 4th node for ID
    print("   1.44 Element Numbering")
    els_sort = np.zeros((nk,4))
    type_sort = np.zeros(nk)
    for ik in range(nk):
        cii = centersG[ik,:]
        test = gmsh.model.mesh.get_elements_by_coordinates(cii[0],cii[1],cii[2],dim=-1,strict=True)
        if len(test)>1:
            print(" - Warning: More than 1 element found for center of element " + str(ik) + " on surface " + str(el_surf[ik]))
        eli = gmsh.model.mesh.get_element_by_coordinates(cii[0],cii[1],cii[2],strict=True)[0]
        if eli in quad_indices:
            els_sort[ik]=quads[np.where(quad_indices==eli)]
            type_sort[ik] = 4
        elif eli in tri_indices:
            els_sort[ik] = np.append(tris[np.where(tri_indices == eli)],10**5+1)
            # els_sort[ik] = np.append(tris[np.where(tri_indices == eli)],tris[np.where(tri_indices == eli)][0][2] )
            type_sort[ik] = 3
    els_sort = els_sort.astype(int)
    type_sort = type_sort.astype(int)
    ELS[0] = els_sort
    ELS[4] = type_sort

    # 4.5 Transformation Matrices
    " 4.5 Output: GEOMA[Ti]: Transformations from local area coordinate Systems to global coordinate system"
    print("   1.45 Global Coordinate Transformation")
    for ia in range(na):
        X = np.array([1,0,0])
        Y = np.array([0, 1, 0])
        Z = np.array([0, 0, 1])

        x = (LS[ia][1]-LS[ia][0])/length(LS[ia][1]-LS[ia][0])
        y = (LS[ia][2]-LS[ia][0])/length(LS[ia][2]-LS[ia][0])
        z = np.cross(x,y)

        GEOMA["T1"][ia] = np.array([[vcos(x,X),vcos(x,Y),vcos(x,Z)],[vcos(y,X),vcos(y,Y),vcos(y,Z)],[vcos(z,X),vcos(z,Y),vcos(z,Z)]])
        GEOMA["T2"][ia] = np.array([[-vcos(y,X),-vcos(y,Y),-vcos(y,Z)],[vcos(x,X),vcos(x,Y),vcos(x,Z)],[vcos(z,X),vcos(z,Y),vcos(z,Z)]])
        GEOMA["T3"][ia] = np.array([[vcos(z,X)**2,vcos(z,X)*vcos(z,Y),vcos(z,X)*vcos(z,Z)],
                        [vcos(z,Y)*vcos(z,X),vcos(z,Y)**2,vcos(z,Y)*vcos(z,Z)],
                        [vcos(z,Z)*vcos(z,X),vcos(z,Z)*vcos(z,Y),vcos(z,Z)**2]])
        # print(GEOMA["T1"][ia]@np.transpose(GEOMA["T1"][ia]))

    # 4.6 Node information
    " 4.6 Output:   - Global and Local coordinates of all nodes and by area"
    "               - COORD[n]: [n_G, n_Ga, n_L, n_La]"
    "               - ELS[3]: n_L->G"
    print("   1.46 Node Information")
    NODESL = [[]]*na
    NODESLa = [[]]*na
    NODESGa = [[]]*na
    for ia in range(na):
        NODESL_a=np.zeros_like(NODESG)
        for ian in range(nn):
            niG = NODESG[ian,:]
            NODESL_a[ian,:] = (niG-O[ia])@np.linalg.inv(GEOMA["T1"][ia])
        NODESL[ia]=NODESL_a[:,0:2]
        els_in_a = ELS[0][np.where(el_surf == ia)[0],:]
        els_in_a = els_in_a.astype(int)
        els_in_a = els_in_a[np.where(els_in_a<10**5)]
        NODESGa[ia] = NODESG[np.unique(np.ndarray.flatten(els_in_a)),:]
        NODESLa[ia] = NODESL_a[np.unique(np.ndarray.flatten(els_in_a)),:][:,0:2]
        ELS[3][ia] = np.zeros(len(NODESGa[ia][:,0]))
        for ik in range(len(NODESGa[ia][:,0])):
            lenik = len(NODESGa[ia][:,0])
            node_ik = NODESGa[ia][ik,:]
            ikk = np.where((NODESG == node_ik).all(axis=1))
            ikk = int(ikk[0])
            ELS[3][ia][ik] = ikk
        ELS[3][ia] = ELS[3][ia].astype(int)
    COORD["n"] = [NODESG,NODESGa,NODESL,NODESLa]

    # 4.7 Flip Order of Elements arranged clockwise
    " 4.7 Output:   - ELS[0] counter clockwise for correct integration"
    print("   1.47 Flip Element Arrangement Clockwise")
    for k in range(nk):
        el = ELS[0][k,:]
        a_k = el_surf[k]
        n1 = NODESL[a_k][el[0],:]
        n2 = NODESL[a_k][el[1],:]
        n3 = NODESL[a_k][el[2], :]
        v1 = n2-n1
        normal1 = [-v1[1],v1[0]]
        v2 = n3-n1
        dir = dotproduct(normal1,v2)
        # check that dir has dimensions --> if dim = 0 then v1 and v2 have the same direction!
        if abs(dir) > 1:
            if dir < 0:
                if ELS[4][k] == 4:
                    ELS[0][k,:]=[el[3],el[2],el[1],el[0]]
                elif ELS[4][k] == 3:
                    ELS[0][k, :] = [el[2], el[1], el[0], el[3]]
        else:
            n1 = NODESL[a_k][el[1], :]
            n2 = NODESL[a_k][el[2], :]
            n3 = NODESL[a_k][el[3], :]
            v1 = n2 - n1
            normal1 = [-v1[1], v1[0]]
            v2 = n3 - n1
            dir = dotproduct(normal1, v2)
            if dir < 0:
                if ELS[4][k] == 4:
                    ELS[0][k, :] = [el[3], el[2], el[1], el[0]]
                elif ELS[4][k] == 3:
                    ELS[0][k, :] = [el[2], el[1], el[0], el[3]]
    # 4.8 Element Center Point Information (2/2) see 4.3
    " 4.8 Output:   - Global and Local coordinates of all element centers and by area"
    "               - COORD[c]: [c_G, c_Ga, c_L, c_La]"
    print("   1.48 Element Center Information 2/2")
    CENTERSL = [[]]*na
    CENTERSLa = [[]]*na
    CENTERSGa = [[]]*na
    for ia in range(na):
        CENTERSL_a = np.zeros_like(centersG)
        for iak in range(nk):
            ciG = centersG[iak,:]
            CENTERSL_a[iak,:] = (ciG-O[ia])@np.linalg.inv(GEOMA["T1"][ia])
        CENTERSL[ia]=CENTERSL_a[:,0:2]
        CENTERSGa[ia] = centersG[np.where(el_surf==ia)]
        CENTERSLa[ia] = CENTERSL_a[np.where(el_surf==ia)][:,0:2]
    COORD["c"] = [centersG,CENTERSGa,CENTERSL,CENTERSLa]

    # 4.9 Element Connectivity Information
    " 4.9 Output:   - ELS[1-2]: [e_Ga, e_La]"
    "               - e_Ga: Connectivity of areas in global node numbering "
    "               - e_La: Connectivity of areas in local node numbering "
    print("   1.49 Element Connectivity")
    for ia in range(na):
        ELS[1][ia] = ELS[0][np.where(el_surf == ia)]
        ELS[2][ia] = ELS[0][np.where(el_surf == ia)]
        for i in range(len(ELS[1][ia][:,0])):
            for j in range(len(ELS[1][ia][0, :])):
                nGij = ELS[1][ia][i][j]
                if nGij < 10**5:
                    ind = np.argwhere(ELS[3][ia]==nGij).flatten()
                    ELS[2][ia][i][j] = ind
                else:
                    ELS[2][ia][i][j] = 10**5+1

    # 4.10 Element Integration Point Information
    " 4.10 Output:  - COORD[ip]: Coordinates of integration points in global and local coordinate systems"
    "               - COORD[ip]: [ip_G, ip_Ga, ip_L, ip_La]"
    print("   1.410 Element Integration Point Information")
    nk = len(ELS[0][:,0])
    IPG = [[]]*nk
    IPGa = [[]]*na
    IPL = [[]]*na
    IPLa = [[]]*na
    go = gauss_order
    for ia in range(na):
        nk_a = len(ELS[2][ia][:,0])
        T1 = GEOMA["T1"][ia]
        for k in range(nk_a):
            el_k = ELS[1][ia][k, :]
            el_k = el_k[el_k < 10**5]
            c_k = COORD["c"][3][ia][k]
            if go == 1:
                [xi, eta, J, J_inv, J_det] = jacobi(k, 0, 0, go, COORD["n"][3][ia], ELS[2][ia])
                # Transformation from uni-element to element shape
                if k == 0:
                    IPLa[ia] = c_k
                    IPGa[ia] = [np.append(c_k, 0) @ T1 + O[ia]]
                else:
                    IPLa[ia] = np.vstack([IPLa[ia], c_k])
                    IPGa[ia] = np.vstack([IPGa[ia], np.append(c_k, 0) @ T1 + O[ia]])
            elif go == 2:
                for i in range(go):
                    for j in range(go):
                        if len(el_k) == 3 and i == 1 and j == 1:
                            continue
                        [xi, eta, J, J_inv, J_det] = jacobi(k, i, j, go, COORD["n"][3][ia], ELS[2][ia])
                        # Transformation from uni-element to element shape
                        if len(el_k) == 3:
                            dxy = [xi-1/3,eta-1/3] @ J
                        else:
                            dxy = [xi, eta] @ J
                        if k == 0 and i == 0 and j == 0:
                            IPLa[ia] = c_k + dxy
                            IPGa[ia] = [np.append(c_k + dxy,0)@T1 + O[ia]]
                        else:
                            IPLa[ia] = np.vstack([IPLa[ia],c_k+dxy])
                            IPGa[ia] = np.vstack([IPGa[ia],np.append(c_k + dxy,0)@T1 + O[ia]])
        if ia < 0.5:
            IPG = IPGa[ia]
        else:
            IPG = np.vstack([IPG,IPGa[ia]])
    COORD["ip"] = [IPG,IPGa,IPLa,IPLa]

    # 4.11 Geometrical Information per Element
    " 4.11 Output:  - GEOMK"
    print("   1.411 Geometrical Information per Element")
    Tk1 = [[]]*nk
    Tk2 = [[]]*nk
    Tk3 = [[]]*nk
    tk = [[]]*nk
    tk2 = [[]]*nk
    nlk = [[]]*nk
    rhoxk = [[]]*nk
    rhoyk = [[]]*nk
    dxk = [[]]*nk
    dyk = [[]]*nk
    sxk = [[]]*nk
    syk = [[]]*nk
    rhopxk = [[]]*nk
    rhopyk = [[]]*nk
    dpxk = [[]]*nk
    dpyk = [[]]*nk
    for k in range(nk):
        a_k = el_surf[k]
        Tk1[k] = GEOMA["T1"][a_k]
        Tk2[k] = GEOMA["T2"][a_k]
        Tk3[k] = GEOMA["T3"][a_k]
        tk[k] = GEOMA["ta"][a_k]
        tk2[k] = GEOMA["ta_2"][a_k]
        nlk[k] = GEOMA["nla"][a_k]
        rhoxk[k] = GEOMA["rhoxal"][a_k]
        rhoyk[k] = GEOMA["rhoyal"][a_k]
        dxk[k] = GEOMA["dxal"][a_k]
        dyk[k] = GEOMA["dyal"][a_k]
        sxk[k] = GEOMA["sxal"][a_k]
        syk[k] = GEOMA["syal"][a_k]
        rhopxk[k] = GEOMA["rhopxal"][a_k]
        rhopyk[k] = GEOMA["rhopyal"][a_k]
        dpxk[k] = GEOMA["dpxal"][a_k]
        dpyk[k] = GEOMA["dpyal"][a_k]
    GEOMK = {"ak"         :el_surf,
            "Tk1"          :Tk1,
            "Tk2"          : Tk2,
            "Tk3"          : Tk3,
            "t"            : tk,
            "t2"           : tk2,
            "nlk"         : nlk,
            "rhox"        : rhoxk,
            "rhoy"        : rhoyk,
            "dx"          : dxk,
            "dy"          : dyk,
            "sx"          : sxk,
            "sy"          : syk,
            "rhopx"     : rhopxk,
            "rhopy"     : rhopyk,
            "dpx"       : dpxk,
            "dpy"       : dpyk,
            }

    # 4.12 Material Information per Element
    " 4.12 Output:  - MATK"
    print("   1.412 Material Information per Element")

    Ek = [[]]*nk
    Ek2 = [[]]*nk
    Gk = [[]]*nk
    vk = [[]]*nk
    vk2 = [[]]*nk
    fcpk = [[]]*nk
    fctk = [[]]*nk
    tb0k = [[]]*nk
    tb1k = [[]]*nk
    tbp0k = [[]]*nk
    tbp1k = [[]]*nk
    ebp1k = [[]]*nk
    ec0k = [[]]*nk
    Dmaxk = [[]]*nk
    Esxk = [[]]*nk
    Eshxk = [[]]*nk
    fsyxk = [[]]*nk
    fpuxk = [[]]*nk
    fsuxk = [[]]*nk
    Esyk = [[]]*nk
    Eshyk = [[]]*nk
    fsyyk = [[]]*nk
    fsuyk = [[]]*nk
    fpuyk = [[]]*nk
    Epxk = [[]]*nk
    Epyk = [[]]*nk
    cmk = [[]]*nk
    phik = [[]]*nk
    ecsk = [[]]*nk
    bk = [[]]*nk
    lk = [[]]*nk
    for k in range(nk):
        a_k = el_surf[k]
        Ek[k] = MATA["Eca"][a_k]
        Ek2[k] = MATA["Eca2"][a_k]
        Gk[k] = MATA["Gca"][a_k]
        vk[k] = MATA["vca"][a_k]
        vk2[k] = MATA["vca2"][a_k]
        fcpk[k] = MATA["fcpa"][a_k]
        fctk[k] = MATA["fcta"][a_k]
        tb0k[k] = MATA["tb0a"][a_k]
        tb1k[k] = MATA["tb1a"][a_k]
        tbp0k[k] = MATA["tbp0a"][a_k]
        tbp1k[k] = MATA["tbp1a"][a_k]
        ebp1k[k] = MATA["ebp1a"][a_k]
        ec0k[k] = MATA["ec0a"][a_k]
        Dmaxk[k] = MATA["Dmaxa"][a_k]
        Esxk[k] = MATA["Esxa"][a_k]
        Epxk[k] = MATA["Epxa"][a_k]
        Eshxk[k] = MATA["Eshxa"][a_k]
        fsyxk[k] = MATA["fsyxa"][a_k]
        fsuxk[k] = MATA["fsuxa"][a_k]
        fpuxk[k] = MATA["fpuxa"][a_k]
        Esyk[k] = MATA["Esya"][a_k]
        Epyk[k] = MATA["Epya"][a_k]
        Eshyk[k] = MATA["Eshya"][a_k]
        fsyyk[k] = MATA["fsyya"][a_k]
        fsuyk[k] = MATA["fsuya"][a_k]
        fpuyk[k] = MATA["fpuya"][a_k]
        cmk[k] = MATA["cma"][a_k]
        phik[k] = MATA["phia"][a_k]
        ecsk[k] = MATA["ecsa"][a_k]
        bk[k] = ms
        lk[k] = ms
    MATK = {"Ec"        : Ek,
            "Ec2"       : Ek2,
            "Gc"        : Gk,
            "vc"        : vk,
            "vc2"       : vk2,
            "fcp"      : fcpk,
            "fct"      : fctk,
            "tb0"      : tb0k,
            "tb1"      : tb1k,
            "tbp0"      : tbp0k,
            "tbp1"      : tbp1k,
            "ebp1"      : ebp1k,
            "ec0"      : ec0k,
            "Dmax"     : Dmaxk,
            "Esx"      : Esxk,
            "Eshx"     : Eshxk,
            "fsyx"     : fsyxk,
            "fsux"     : fsuxk,
            "fpux"     : fpuxk,
            "Esy"      : Esyk,
            "Eshy"     : Eshyk,
            "fsyy"     : fsyyk,
            "fsuy"     : fsuyk,
            "fpuy"     : fpuyk,
            "Epx"       : Epxk,
            "Epy"       : Epyk,
            "cm"        :cmk,
            "phi"     : phik,
            "ecs"     : ecsk,
            "ms"      : ms}

    # 4.13 Masks for nodes in areas
    " 4.13 Output:  - MASK: bool if node is in area"
    print("   1.413 Masks for Nodes in Areas")
    MASK = {}
    for ia in range(na):
        count = 0
        MASK[ia]=[False]*len(NODESG[:,0])
        for row in NODESG:
            MASK[ia][count] = row.tolist() in COORD["n"][1][ia].tolist()
            count+=1

    # 4.14  Coplanar Nodes
    " 4.14 Output:  - copln: vector assigning for each node, whether coplanar (1) or not (0)"
    print("   1.414 Coplanar Nodes")
    numn = len(NODESG[:,1])
    copln = np.zeros((numn,1))
    for n in range(numn):
        copln[n] = find_copl(n, ELS, Tk1)

    return MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln

